/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2024 Stanford University and the Authors.           *
 * Authors: OpenMM Contributors                                               *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a   *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,  *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

/**
 * Tests the partial setPositions(positions, firstIndex) and
 * setVelocities(velocities, firstIndex) overloads on the Context class.
 */

#include "ReferenceTests.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

/**
 * Build a minimal system of N free particles (no forces) and return a Context.
 * A HarmonicBondForce with no bonds is added to satisfy platforms that require
 * at least one Force object.
 */
static void makeSystem(int numParticles, System& system) {
    system.addForce(new HarmonicBondForce());
    for (int i = 0; i < numParticles; ++i)
        system.addParticle(1.0);
}

/** Set a single particle's position and verify the rest are unchanged. */
void testSinglePositionUpdate() {
    const int N = 10;
    System system;
    makeSystem(N, system);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    vector<Vec3> positions(N);
    for (int i = 0; i < N; ++i)
        positions[i] = Vec3(i * 0.1, 0.0, 0.0);
    context.setPositions(positions);

    // Update only particle 3.
    vector<Vec3> update = {Vec3(9.9, 8.8, 7.7)};
    context.setPositions(update, 3);

    State state = context.getState(State::Positions);
    const vector<Vec3>& got = state.getPositions();
    ASSERT_EQUAL_VEC(Vec3(9.9, 8.8, 7.7), got[3], TOL);
    for (int i = 0; i < N; ++i)
        if (i != 3)
            ASSERT_EQUAL_VEC(Vec3(i * 0.1, 0.0, 0.0), got[i], TOL);
}

/** Update a contiguous block of positions (GCMC water-molecule pattern). */
void testBlockPositionUpdate() {
    const int N = 20;
    System system;
    makeSystem(N, system);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    vector<Vec3> positions(N);
    for (int i = 0; i < N; ++i)
        positions[i] = Vec3(i * 0.1, 0.0, 0.0);
    context.setPositions(positions);

    // Update particles 7, 8, 9 (like replacing a TIP3P water).
    vector<Vec3> update = {
        Vec3(1.0, 2.0, 3.0),
        Vec3(1.1, 2.1, 3.1),
        Vec3(1.2, 2.2, 3.2)
    };
    context.setPositions(update, 7);

    State state = context.getState(State::Positions);
    const vector<Vec3>& got = state.getPositions();
    ASSERT_EQUAL_VEC(Vec3(1.0, 2.0, 3.0), got[7], TOL);
    ASSERT_EQUAL_VEC(Vec3(1.1, 2.1, 3.1), got[8], TOL);
    ASSERT_EQUAL_VEC(Vec3(1.2, 2.2, 3.2), got[9], TOL);
    for (int i = 0; i < N; ++i)
        if (i < 7 || i > 9)
            ASSERT_EQUAL_VEC(Vec3(i * 0.1, 0.0, 0.0), got[i], TOL);
}

/** Update at firstIndex == 0 and firstIndex == N-1 (boundary cases). */
void testBoundaryPositionUpdates() {
    const int N = 5;
    System system;
    makeSystem(N, system);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    vector<Vec3> positions(N, Vec3(0.0, 0.0, 0.0));
    context.setPositions(positions);

    // Update first particle.
    context.setPositions(vector<Vec3>{Vec3(1.0, 1.0, 1.0)}, 0);
    // Update last particle.
    context.setPositions(vector<Vec3>{Vec3(2.0, 2.0, 2.0)}, N - 1);

    State state = context.getState(State::Positions);
    const vector<Vec3>& got = state.getPositions();
    ASSERT_EQUAL_VEC(Vec3(1.0, 1.0, 1.0), got[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(2.0, 2.0, 2.0), got[N - 1], TOL);
    for (int i = 1; i < N - 1; ++i)
        ASSERT_EQUAL_VEC(Vec3(0.0, 0.0, 0.0), got[i], TOL);
}

/** Empty update (zero-length positions vector) is a no-op. */
void testEmptyPositionUpdate() {
    const int N = 5;
    System system;
    makeSystem(N, system);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    vector<Vec3> positions(N);
    for (int i = 0; i < N; ++i)
        positions[i] = Vec3(i * 0.5, 0.0, 0.0);
    context.setPositions(positions);
    context.setPositions(vector<Vec3>{}, 0);   // no-op

    State state = context.getState(State::Positions);
    const vector<Vec3>& got = state.getPositions();
    for (int i = 0; i < N; ++i)
        ASSERT_EQUAL_VEC(Vec3(i * 0.5, 0.0, 0.0), got[i], TOL);
}

/** Out-of-bounds firstIndex should throw. */
void testOutOfBoundsPositionUpdate() {
    const int N = 5;
    System system;
    makeSystem(N, system);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    vector<Vec3> positions(N, Vec3(0.0, 0.0, 0.0));
    context.setPositions(positions);

    bool threw = false;
    try {
        context.setPositions(vector<Vec3>{Vec3(1.0, 1.0, 1.0)}, N); // one past the end
    } catch (const OpenMMException&) {
        threw = true;
    }
    ASSERT(threw);
}

/** Single velocity update. */
void testSingleVelocityUpdate() {
    const int N = 10;
    System system;
    makeSystem(N, system);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    vector<Vec3> positions(N, Vec3(0.0, 0.0, 0.0));
    context.setPositions(positions);

    vector<Vec3> velocities(N);
    for (int i = 0; i < N; ++i)
        velocities[i] = Vec3(i * 0.01, 0.0, 0.0);
    context.setVelocities(velocities);

    // Update only particle 5.
    vector<Vec3> update = {Vec3(9.0, 8.0, 7.0)};
    context.setVelocities(update, 5);

    State state = context.getState(State::Velocities);
    const vector<Vec3>& got = state.getVelocities();
    ASSERT_EQUAL_VEC(Vec3(9.0, 8.0, 7.0), got[5], TOL);
    for (int i = 0; i < N; ++i)
        if (i != 5)
            ASSERT_EQUAL_VEC(Vec3(i * 0.01, 0.0, 0.0), got[i], TOL);
}

/** Block velocity update. */
void testBlockVelocityUpdate() {
    const int N = 20;
    System system;
    makeSystem(N, system);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    vector<Vec3> positions(N, Vec3(0.0, 0.0, 0.0));
    context.setPositions(positions);
    vector<Vec3> velocities(N, Vec3(0.0, 0.0, 0.0));
    context.setVelocities(velocities);

    vector<Vec3> update = {
        Vec3(0.5, 0.6, 0.7),
        Vec3(0.8, 0.9, 1.0),
        Vec3(1.1, 1.2, 1.3)
    };
    context.setVelocities(update, 12);

    State state = context.getState(State::Velocities);
    const vector<Vec3>& got = state.getVelocities();
    ASSERT_EQUAL_VEC(Vec3(0.5, 0.6, 0.7), got[12], TOL);
    ASSERT_EQUAL_VEC(Vec3(0.8, 0.9, 1.0), got[13], TOL);
    ASSERT_EQUAL_VEC(Vec3(1.1, 1.2, 1.3), got[14], TOL);
    for (int i = 0; i < N; ++i)
        if (i < 12 || i > 14)
            ASSERT_EQUAL_VEC(Vec3(0.0, 0.0, 0.0), got[i], TOL);
}

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testSinglePositionUpdate();
        testBlockPositionUpdate();
        testBoundaryPositionUpdates();
        testEmptyPositionUpdate();
        testOutOfBoundsPositionUpdate();
        testSingleVelocityUpdate();
        testBlockVelocityUpdate();
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
