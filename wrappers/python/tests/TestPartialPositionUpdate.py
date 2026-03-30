"""
Tests for Context.setPositions(positions, firstIndex) and
Context.setVelocities(velocities, firstIndex).

The correctness tests run on the Reference platform and cover:
  - single-particle updates
  - block updates (consecutive particles, e.g. a TIP3P water)
  - boundary positions (firstIndex == 0, firstIndex == N-1)
  - empty update (no-op)
  - out-of-bounds rejection

The performance test (test_partial_update_performance) runs on the default
platform and prints a timing comparison between the full and partial APIs.
It does not make a hard timing assertion, but records the measurements so
they are visible in the test output.
"""

import time
import unittest
import openmm as mm
from openmm import unit


def _make_context(n_particles, platform_name="Reference"):
    """Return a Context for n_particles free particles on the named platform."""
    system = mm.System()
    # Add a HarmonicBondForce with no bonds to satisfy platforms that require
    # at least one Force.
    system.addForce(mm.HarmonicBondForce())
    for _ in range(n_particles):
        system.addParticle(1.0)
    integrator = mm.VerletIntegrator(0.001)
    platform = mm.Platform.getPlatform(platform_name)
    return mm.Context(system, integrator, platform)


class TestPartialPositionUpdateCorrectness(unittest.TestCase):
    """Correctness tests for setPositions(positions, firstIndex)."""

    N = 20
    platform_name = "Reference"

    def setUp(self):
        self.context = _make_context(self.N, self.platform_name)
        self.initial = [mm.Vec3(i * 0.1, 0.0, 0.0) for i in range(self.N)]
        self.context.setPositions(self.initial)

    def _get_positions(self):
        return (
            self.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer)
        )

    def _get_velocities(self):
        return (
            self.context.getState(getVelocities=True)
            .getVelocities(asNumpy=True)
            .value_in_unit(unit.nanometer / unit.picosecond)
        )

    # ------------------------------------------------------------------
    # Position tests
    # ------------------------------------------------------------------

    def test_single_particle(self):
        """Updating one particle leaves all others unchanged."""
        self.context.setPositions([mm.Vec3(9.9, 8.8, 7.7)], 3)
        pos = self._get_positions()
        self.assertAlmostEqual(pos[3][0], 9.9, places=5)
        self.assertAlmostEqual(pos[3][1], 8.8, places=5)
        self.assertAlmostEqual(pos[3][2], 7.7, places=5)
        for i in range(self.N):
            if i != 3:
                self.assertAlmostEqual(
                    pos[i][0],
                    i * 0.1,
                    places=5,
                    msg=f"Particle {i} was unexpectedly modified",
                )

    def test_block_update(self):
        """Updating a block of 3 consecutive particles (TIP3P-like GCMC move)."""
        new_pos = [
            mm.Vec3(1.0, 2.0, 3.0),
            mm.Vec3(1.1, 2.1, 3.1),
            mm.Vec3(1.2, 2.2, 3.2),
        ]
        first = 7
        self.context.setPositions(new_pos, first)
        pos = self._get_positions()
        for k, p in enumerate(new_pos):
            idx = first + k
            self.assertAlmostEqual(pos[idx][0], p[0], places=5)
            self.assertAlmostEqual(pos[idx][1], p[1], places=5)
            self.assertAlmostEqual(pos[idx][2], p[2], places=5)
        for i in range(self.N):
            if i < first or i >= first + len(new_pos):
                self.assertAlmostEqual(
                    pos[i][0],
                    i * 0.1,
                    places=5,
                    msg=f"Particle {i} was unexpectedly modified",
                )

    def test_first_particle(self):
        """firstIndex == 0 (boundary case)."""
        self.context.setPositions([mm.Vec3(5.5, 5.5, 5.5)], 0)
        pos = self._get_positions()
        self.assertAlmostEqual(pos[0][0], 5.5, places=5)
        for i in range(1, self.N):
            self.assertAlmostEqual(pos[i][0], i * 0.1, places=5)

    def test_last_particle(self):
        """firstIndex == N-1 (boundary case)."""
        self.context.setPositions([mm.Vec3(5.5, 5.5, 5.5)], self.N - 1)
        pos = self._get_positions()
        self.assertAlmostEqual(pos[self.N - 1][0], 5.5, places=5)
        for i in range(self.N - 1):
            self.assertAlmostEqual(pos[i][0], i * 0.1, places=5)

    def test_empty_update_is_noop(self):
        """An empty positions list is a no-op."""
        self.context.setPositions([], 0)
        pos = self._get_positions()
        for i in range(self.N):
            self.assertAlmostEqual(pos[i][0], i * 0.1, places=5)

    def test_result_matches_full_setpositions(self):
        """Partial update produces identical state to building the full array."""
        import random

        random.seed(42)
        first = 4
        new_pos = [
            mm.Vec3(random.random(), random.random(), random.random()) for _ in range(5)
        ]

        # Approach 1: use partial update.
        ctx1 = _make_context(self.N, self.platform_name)
        ctx1.setPositions(self.initial)
        ctx1.setPositions(new_pos, first)

        # Approach 2: rebuild the full array and use the original API.
        ctx2 = _make_context(self.N, self.platform_name)
        full = list(self.initial)
        for k, p in enumerate(new_pos):
            full[first + k] = p
        ctx2.setPositions(full)

        pos1 = ctx1.getState(getPositions=True).getPositions()
        pos2 = ctx2.getState(getPositions=True).getPositions()
        for i in range(self.N):
            for d in range(3):
                self.assertAlmostEqual(
                    pos1[i][d],
                    pos2[i][d],
                    places=5,
                    msg=f"Mismatch at particle {i} dim {d}",
                )

    def test_out_of_bounds_raises(self):
        """firstIndex past the end raises an exception."""
        with self.assertRaises(Exception):
            self.context.setPositions([mm.Vec3(1.0, 1.0, 1.0)], self.N)

    def test_overflow_raises(self):
        """firstIndex + len(positions) > N raises an exception."""
        with self.assertRaises(Exception):
            self.context.setPositions(
                [mm.Vec3(1.0, 1.0, 1.0), mm.Vec3(2.0, 2.0, 2.0)], self.N - 1
            )

    def test_with_units(self):
        """Positions supplied with OpenMM units are correctly stripped."""
        new_pos = unit.Quantity([mm.Vec3(0.5, 0.5, 0.5)], unit.nanometer)
        self.context.setPositions(new_pos, 2)
        pos = self._get_positions()
        self.assertAlmostEqual(pos[2][0], 0.5, places=5)

    # ------------------------------------------------------------------
    # Velocity tests
    # ------------------------------------------------------------------

    def _init_velocities(self):
        vels = [mm.Vec3(i * 0.01, 0.0, 0.0) for i in range(self.N)]
        self.context.setVelocities(vels)
        return vels

    def test_single_velocity_update(self):
        """Updating one particle's velocity leaves all others unchanged."""
        init_vels = self._init_velocities()
        self.context.setVelocities([mm.Vec3(9.0, 8.0, 7.0)], 5)
        vels = self._get_velocities()
        self.assertAlmostEqual(vels[5][0], 9.0, places=5)
        self.assertAlmostEqual(vels[5][1], 8.0, places=5)
        self.assertAlmostEqual(vels[5][2], 7.0, places=5)
        for i in range(self.N):
            if i != 5:
                self.assertAlmostEqual(
                    vels[i][0],
                    i * 0.01,
                    places=5,
                    msg=f"Velocity of particle {i} was unexpectedly modified",
                )

    def test_block_velocity_update(self):
        """Updating a block of velocities."""
        self._init_velocities()
        new_vels = [
            mm.Vec3(0.5, 0.6, 0.7),
            mm.Vec3(0.8, 0.9, 1.0),
            mm.Vec3(1.1, 1.2, 1.3),
        ]
        first = 12
        self.context.setVelocities(new_vels, first)
        vels = self._get_velocities()
        for k, v in enumerate(new_vels):
            idx = first + k
            self.assertAlmostEqual(vels[idx][0], v[0], places=5)
            self.assertAlmostEqual(vels[idx][1], v[1], places=5)
            self.assertAlmostEqual(vels[idx][2], v[2], places=5)

    def test_velocity_with_units(self):
        """Velocities supplied with OpenMM units are correctly stripped."""
        self._init_velocities()
        new_vel = unit.Quantity(
            [mm.Vec3(1.0, 0.0, 0.0)], unit.nanometer / unit.picosecond
        )
        self.context.setVelocities(new_vel, 3)
        vels = self._get_velocities()
        self.assertAlmostEqual(vels[3][0], 1.0, places=5)


class TestPartialPositionUpdatePerformance(unittest.TestCase):
    """
    Timing comparison between setPositions(all_N) and setPositions(k, firstIndex).

    The test asserts that the partial call is faster, but prints the raw
    numbers so they are visible in CI output.  The benefit is largest at the
    Python layer (avoiding the O(N) SWIG conversion), so N should be large.
    """

    N = 10_000
    K = 3  # atoms changed per GCMC move (TIP3P)
    N_ITER = 200  # repetitions for each timing measurement

    @classmethod
    def setUpClass(cls):
        cls.context = _make_context(cls.N, "Reference")
        cls.all_positions = [mm.Vec3(i * 0.001, 0.0, 0.0) for i in range(cls.N)]
        cls.context.setPositions(cls.all_positions)
        cls.partial_positions = [mm.Vec3(0.5, 0.5, 0.5) for _ in range(cls.K)]

    def _time(self, fn, n_iter):
        # Warm up.
        for _ in range(max(1, n_iter // 10)):
            fn()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        return (time.perf_counter() - t0) / n_iter

    def test_partial_update_is_faster(self):
        t_full = self._time(
            lambda: self.context.setPositions(self.all_positions), self.N_ITER
        )
        t_partial = self._time(
            lambda: self.context.setPositions(self.partial_positions, 0), self.N_ITER
        )
        print(
            f"\n  setPositions({self.N} atoms):  {t_full * 1e3:.3f} ms/call"
            f"\n  setPositions({self.K} atoms, firstIndex=0): {t_partial * 1e3:.3f} ms/call"
            f"\n  Speedup: {t_full / t_partial:.1f}x"
        )
        self.assertLess(
            t_partial,
            t_full,
            "Partial setPositions was not faster than the full update",
        )

    def test_partial_velocity_update_is_faster(self):
        all_velocities = [mm.Vec3(i * 0.0001, 0.0, 0.0) for i in range(self.N)]
        self.context.setVelocities(all_velocities)
        partial_velocities = [mm.Vec3(0.1, 0.0, 0.0) for _ in range(self.K)]

        t_full = self._time(
            lambda: self.context.setVelocities(all_velocities), self.N_ITER
        )
        t_partial = self._time(
            lambda: self.context.setVelocities(partial_velocities, 0), self.N_ITER
        )
        print(
            f"\n  setVelocities({self.N} atoms):  {t_full * 1e3:.3f} ms/call"
            f"\n  setVelocities({self.K} atoms, firstIndex=0): {t_partial * 1e3:.3f} ms/call"
            f"\n  Speedup: {t_full / t_partial:.1f}x"
        )
        self.assertLess(
            t_partial,
            t_full,
            "Partial setVelocities was not faster than the full update",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
