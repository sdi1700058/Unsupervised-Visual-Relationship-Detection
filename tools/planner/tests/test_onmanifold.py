#!/usr/bin/env python3
"""Tests for the on-manifold planner used as H14's ablation."""

import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class TestObservedGraph(unittest.TestCase):

    def test_a_self_loop_is_dropped(self):
        """Repeated consecutive latents are a no-op, not a free step."""
        from tools.planner.onmanifold import observed_graph

        z = np.array([[0, 0], [0, 0], [1, 0]], dtype=np.int8)
        index, adj, nodes = observed_graph(z)
        self.assertEqual(len(nodes), 2)
        self.assertNotIn(index[z[0].tobytes()], adj[index[z[0].tobytes()]])

    def test_edges_are_only_observed_transitions(self):
        from tools.planner.onmanifold import observed_graph

        # 0->1->2 observed; 0->2 is NOT, even though both states exist.
        z = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int8)
        index, adj, _ = observed_graph(z)
        a, c = index[z[0].tobytes()], index[z[2].tobytes()]
        self.assertNotIn(c, adj[a])


class TestPath(unittest.TestCase):

    def test_the_path_never_leaves_the_observed_set(self):
        """The property the whole experiment rests on."""
        from tools.planner.onmanifold import shortest_observed_path

        rng = np.random.RandomState(0)
        z = np.cumsum(rng.randint(0, 2, size=(30, 8)), axis=0).astype(np.int8) % 2
        obs = {r.tobytes() for r in z}
        path = shortest_observed_path(z, 0, 20)
        self.assertIsNotNone(path)
        for state in path:
            self.assertIn(state.astype(np.int8).tobytes(), obs)

    def test_it_finds_a_shortcut_through_a_repeated_state(self):
        """If a latent recurs, the observed graph has a genuine shortcut."""
        from tools.planner.onmanifold import shortest_observed_path

        # 0 -> 1 -> 2 -> 1 -> 3 : reaching 3 from 0 takes 2 steps, not 4.
        z = np.array([[0, 0], [1, 0], [1, 1], [1, 0], [0, 1]], dtype=np.int8)
        path = shortest_observed_path(z, 0, 4)
        self.assertIsNotNone(path)
        self.assertEqual(len(path) - 1, 2)

    def test_an_unreachable_goal_returns_none_rather_than_a_guess(self):
        from tools.planner.onmanifold import shortest_observed_path

        z = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int8)
        # Frame 2 is reachable from 1, but nothing leads back to frame 0.
        self.assertIsNone(shortest_observed_path(z, 2, 0))

    def test_identical_endpoints_need_no_steps(self):
        from tools.planner.onmanifold import shortest_observed_path

        z = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int8)
        path = shortest_observed_path(z, 0, 2)
        self.assertEqual(len(path), 1)


if __name__ == "__main__":
    unittest.main()
