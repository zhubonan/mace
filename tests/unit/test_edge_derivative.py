import torch
from torch.testing import assert_close

from mace.modules import models
from mace.modules.utils import get_symmetric_displacement
from mace.tools.edge_derivative import (
    assemble_edge_forces,
    assemble_edge_virials_stress,
    trace_edge_derivative_outputs,
)


def test_edge_force_assembly_matches_position_gradient_with_duplicates():
    positions = torch.tensor(
        [[0.1, -0.2, 0.3], [1.2, 0.4, -0.7], [-0.3, 1.1, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 0], [1, 0, 2, 1, 1]], dtype=torch.long
    )
    sender, receiver = edge_index
    vectors = positions[receiver] - positions[sender]
    energy = (
        0.4 * vectors.square().sum(dim=-1)
        + 0.1 * vectors[:, 0] * vectors[:, 1]
    ).sum()
    position_gradient, edge_gradient = torch.autograd.grad(
        energy, (positions, vectors), retain_graph=True
    )

    actual = assemble_edge_forces(
        edge_gradient, edge_index, num_atoms=positions.shape[0]
    )
    assert_close(actual, -position_gradient)
    assert_close(actual.sum(dim=0), torch.zeros(3, dtype=actual.dtype))


def test_edge_virial_stress_matches_periodic_displacement_gradient():
    positions = torch.tensor(
        [
            [0.2, 0.4, 0.7],
            [1.4, 0.3, 0.1],
            [0.1, 1.1, 0.5],
            [1.2, 1.8, 0.9],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    cell = torch.tensor(
        [
            [[3.2, 0.1, 0.0], [0.4, 2.9, 0.2], [0.1, 0.3, 3.4]],
            [[2.7, 0.2, 0.1], [0.3, 3.1, 0.4], [0.0, 0.2, 2.8]],
        ],
        dtype=torch.float64,
    )
    node_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 0, 1, 2, 3, 2, 3], [1, 0, 1, 0, 3, 2, 3, 2]],
        dtype=torch.long,
    )
    unit_shifts = torch.tensor(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, -1, 0],
        ],
        dtype=torch.float64,
    )
    displacement = torch.zeros(
        (2, 3, 3), dtype=torch.float64, requires_grad=True
    )
    deformed_positions, shifts, displacement = get_symmetric_displacement(
        positions=positions,
        unit_shifts=unit_shifts,
        cell=cell,
        edge_index=edge_index,
        num_graphs=2,
        batch=node_batch,
        displacement=displacement,
    )
    vectors = (
        deformed_positions[edge_index[1]]
        - deformed_positions[edge_index[0]]
        + shifts
    )
    energy = (0.2 * vectors.square().sum(dim=-1) + vectors[:, 0] ** 3).sum()
    position_gradient, displacement_gradient, edge_gradient = torch.autograd.grad(
        energy, (positions, displacement, vectors), retain_graph=True
    )

    forces = assemble_edge_forces(edge_gradient, edge_index, positions.shape[0])
    virials, stress = assemble_edge_virials_stress(
        edge_gradient,
        vectors,
        edge_index,
        node_batch,
        cell,
        num_graphs=2,
        compute_stress=True,
    )
    assert_close(forces, -position_gradient)
    assert_close(forces.sum(dim=0), torch.zeros(3, dtype=forces.dtype))
    assert_close(virials, -displacement_gradient)
    volume = torch.linalg.det(cell).abs().view(-1, 1, 1)
    assert_close(stress, displacement_gradient / volume)


def test_edge_derivative_trace_context_restores_model_globals():
    original_prepare = models.prepare_graph
    original_outputs = models.get_outputs
    with trace_edge_derivative_outputs():
        assert models.prepare_graph is not original_prepare
        assert models.get_outputs is not original_outputs
    assert models.prepare_graph is original_prepare
    assert models.get_outputs is original_outputs


def test_edge_derivative_trace_context_restores_model_globals_after_error():
    original_prepare = models.prepare_graph
    original_outputs = models.get_outputs
    try:
        with trace_edge_derivative_outputs():
            raise RuntimeError("trace failure")
    except RuntimeError as error:
        assert str(error) == "trace failure"
    assert models.prepare_graph is original_prepare
    assert models.get_outputs is original_outputs
