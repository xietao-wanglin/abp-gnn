import torch
import unittest
from torch_geometric.data import Data
from src.models.egnn import EGNN


class TestEquivariance(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 5
        self.in_node_nf = 16
        self.in_edge_nf = 4
        self.hidden_nf = 32
        self.out_node_nf = 16

        self.model = EGNN(
            n_layers=3,
            in_node_nf=self.in_node_nf,
            in_edge_nf=self.in_edge_nf,
            hidden_nf=self.hidden_nf,
            out_node_nf=self.out_node_nf,
            norm=True,
        )
        self.model.eval()

    def test_egnn(self):
        x = torch.randn(self.n_nodes, 2)
        box_length = None
        h = torch.randn(self.n_nodes, self.in_node_nf)
        theta = torch.rand(self.n_nodes, 1) * 2 * torch.pi

        adj = torch.ones((self.n_nodes, self.n_nodes)) - torch.eye(self.n_nodes)
        edge_index = adj.nonzero().T
        edge_attr = torch.randn(edge_index.shape[1], self.in_edge_nf)

        data = Data(
            h=h,
            x=x,
            theta=theta,
            edge_index=edge_index,
            edge_attr=edge_attr,
            box_length=box_length,
        )
        h_out, x_out, theta_out = self.model(data)

        alpha = torch.rand(1) * 2 * torch.pi
        print(f"Rotation: {alpha.item()}")
        c = torch.tensor([1.5, -2.1])

        R = torch.tensor(
            [
                [torch.cos(alpha), -torch.sin(alpha)],
                [torch.sin(alpha), torch.cos(alpha)],
            ]
        )

        x_trans = (x @ R.T) + c
        theta_trans = theta + alpha

        data_trans = Data(
            h=h,
            x=x_trans,
            theta=theta_trans,
            edge_index=edge_index,
            edge_attr=edge_attr,
            box_length=box_length,
        )

        h_out_trans, x_out_trans, theta_out_trans = self.model(data_trans)

        expected_x_out = (x_out @ R.T) + c
        expected_theta_out = theta_out + alpha

        self.assertTrue(
            torch.isclose(h_out_trans, h_out, atol=1e-5, rtol=1e-5).all(),
            "Node feature invariance failed",
        )

        self.assertTrue(
            torch.isclose(
                theta_out_trans, expected_theta_out, atol=1e-5, rtol=1e-5
            ).all(),
            "Theta equivariance failed",
        )

        self.assertTrue(
            torch.isclose(x_out_trans, expected_x_out, atol=1e-5, rtol=1e-5).all(),
            "Position equivariance failed",
        )

    def test_egnn_pbc(self):
        x = torch.randn(self.n_nodes, 2)
        box_length = torch.tensor(2).unsqueeze(0)
        h = torch.randn(self.n_nodes, self.in_node_nf)
        theta = torch.rand(self.n_nodes, 1) * 2 * torch.pi

        adj = torch.ones((self.n_nodes, self.n_nodes)) - torch.eye(self.n_nodes)
        edge_index = adj.nonzero().T
        edge_attr = torch.randn(edge_index.shape[1], self.in_edge_nf)

        data = Data(
            h=h,
            x=x,
            theta=theta,
            edge_index=edge_index,
            edge_attr=edge_attr,
            box_length=box_length,
        )
        h_out, x_out, theta_out = self.model(data)

        alpha = torch.rand(1) * 2 * torch.pi
        print(f"Rotation: {alpha.item()}")
        c = torch.tensor([1.5, -2.1])

        R = torch.tensor(
            [
                [torch.cos(alpha), -torch.sin(alpha)],
                [torch.sin(alpha), torch.cos(alpha)],
            ]
        )

        x_trans = (x @ R.T) + c
        theta_trans = theta + alpha

        data_trans = Data(
            h=h,
            x=x_trans,
            theta=theta_trans,
            edge_index=edge_index,
            edge_attr=edge_attr,
            box_length=box_length,
        )

        h_out_trans, x_out_trans, theta_out_trans = self.model(data_trans)

        expected_x_out = (x_out @ R.T) + c
        expected_theta_out = theta_out + alpha

        self.assertTrue(
            torch.isclose(h_out_trans, h_out, atol=1e-5, rtol=1e-5).all(),
            "Node feature invariance failed",
        )

        self.assertTrue(
            torch.isclose(
                theta_out_trans, expected_theta_out, atol=1e-5, rtol=1e-5
            ).all(),
            "Theta equivariance failed",
        )

        self.assertTrue(
            not torch.isclose(x_out_trans, expected_x_out, atol=1e-5, rtol=1e-5).all(),
            "Position equivariance did NOT fail",
        )

    def test_egnn_normal_pbc(self):
        x = torch.randn(self.n_nodes, 2)
        box_length = torch.tensor(0.2).unsqueeze(0)
        h = torch.randn(self.n_nodes, self.in_node_nf)
        theta = torch.rand(self.n_nodes, 1) * 2 * torch.pi

        adj = torch.ones((self.n_nodes, self.n_nodes)) - torch.eye(self.n_nodes)
        edge_index = adj.nonzero().T
        edge_attr = torch.randn(edge_index.shape[1], self.in_edge_nf)

        data = Data(
            h=h,
            x=x,
            theta=theta,
            edge_index=edge_index,
            edge_attr=edge_attr,
            box_length=box_length,
        )
        h_out, x_out, theta_out = self.model(data)

        alpha = torch.tensor(torch.pi / 2).unsqueeze(0)
        print(f"Rotation: {alpha.item()}")
        c = torch.tensor([1.5, -2.1])

        R = torch.tensor(
            [
                [torch.cos(alpha), -torch.sin(alpha)],
                [torch.sin(alpha), torch.cos(alpha)],
            ]
        )

        x_trans = (x @ R.T) + c
        theta_trans = theta + alpha

        data_trans = Data(
            h=h,
            x=x_trans,
            theta=theta_trans,
            edge_index=edge_index,
            edge_attr=edge_attr,
            box_length=box_length,
        )

        h_out_trans, x_out_trans, theta_out_trans = self.model(data_trans)

        expected_x_out = (x_out @ R.T) + c
        expected_theta_out = theta_out + alpha

        self.assertTrue(
            torch.isclose(h_out_trans, h_out, atol=1e-5, rtol=1e-5).all(),
            "Node feature invariance failed",
        )

        self.assertTrue(
            torch.isclose(
                theta_out_trans, expected_theta_out, atol=1e-5, rtol=1e-5
            ).all(),
            "Theta equivariance failed",
        )

        self.assertTrue(
            torch.isclose(x_out_trans, expected_x_out, atol=1e-5, rtol=1e-5).all(),
            "Position equivariance failed",
        )
