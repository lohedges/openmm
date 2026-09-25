/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
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

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

/**
 * Build a system containing the three kinds of group the constraint code treats
 * differently: three atom clusters with all three distances constrained, which are
 * handled by SETTLE; a central atom with three peripheral atoms at equal distances,
 * which is handled as a SHAKE cluster; and chains of four atoms, which fall through
 * to CCMA.  The indices of the constraints in each category are returned.
 */
void buildConstrainedSystem(System& system, vector<Vec3>& positions, vector<int>& settle,
        vector<int>& shake, vector<int>& ccma) {
    const int numOfEach = 4;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    auto place = [&] (double spacing) {
        Vec3 base(5.0*genrand_real2(sfmt), 5.0*genrand_real2(sfmt), 5.0*genrand_real2(sfmt));
        return base;
    };

    // Three atom clusters, handled by SETTLE.

    for (int i = 0; i < numOfEach; i++) {
        int first = system.getNumParticles();
        system.addParticle(16.0);
        system.addParticle(1.0);
        system.addParticle(1.0);
        Vec3 base = place(0.1);
        positions.push_back(base);
        positions.push_back(base+Vec3(0.09572, 0, 0));
        positions.push_back(base+Vec3(-0.0239987, 0.0926627, 0));
        settle.push_back(system.addConstraint(first, first+1, 0.09572));
        settle.push_back(system.addConstraint(first, first+2, 0.09572));
        settle.push_back(system.addConstraint(first+1, first+2, 0.15139));
    }

    // A central atom with three peripheral atoms at equal distances, handled as a
    // SHAKE cluster.

    for (int i = 0; i < numOfEach; i++) {
        int first = system.getNumParticles();
        system.addParticle(12.0);
        for (int j = 0; j < 3; j++)
            system.addParticle(1.0);
        Vec3 base = place(0.11);
        positions.push_back(base);
        positions.push_back(base+Vec3(0.11, 0, 0));
        positions.push_back(base+Vec3(0, 0.11, 0));
        positions.push_back(base+Vec3(0, 0, 0.11));
        for (int j = 0; j < 3; j++)
            shake.push_back(system.addConstraint(first, first+1+j, 0.11));
    }

    // Chains of four atoms.  The middle atoms are each involved in two constraints, so
    // these cannot be treated as clusters and fall through to CCMA.

    for (int i = 0; i < numOfEach; i++) {
        int first = system.getNumParticles();
        for (int j = 0; j < 4; j++)
            system.addParticle(12.0);
        Vec3 base = place(0.15);
        for (int j = 0; j < 4; j++)
            positions.push_back(base+Vec3(0.15*j, 0, 0));
        for (int j = 0; j < 3; j++)
            ccma.push_back(system.addConstraint(first+j, first+j+1, 0.15));
    }
    system.setDefaultPeriodicBoxVectors(Vec3(10, 0, 0), Vec3(0, 10, 0), Vec3(0, 0, 10));
}

/**
 * Check that every constraint in the system is satisfied to within a tolerance.
 */
void verifyConstraints(const System& system, Context& context, double tol) {
    context.applyConstraints(1e-6);
    State state = context.getState(State::Positions);
    const vector<Vec3>& pos = state.getPositions();
    for (int i = 0; i < system.getNumConstraints(); i++) {
        int p1, p2;
        double distance;
        system.getConstraintParameters(i, p1, p2, distance);
        Vec3 delta = pos[p1]-pos[p2];
        ASSERT_EQUAL_TOL(distance, sqrt(delta.dot(delta)), tol);
    }
}

void testUpdateConstraints() {
    // Change the distances in a way that leaves the division between the constraint
    // algorithms unchanged, so the update can be applied without rebuilding.

    System system;
    vector<Vec3> positions;
    vector<int> settle, shake, ccma;
    buildConstrainedSystem(system, positions, settle, shake, ccma);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    verifyConstraints(system, context, 1e-4);

    // Scale each SETTLE cluster's three distances together, so the same atom remains the
    // central one, scale each SHAKE cluster's peripheral distances together, so they stay
    // equal, and change the CCMA constraints independently.

    for (int i = 0; i < (int) settle.size(); i++) {
        int p1, p2;
        double distance;
        system.getConstraintParameters(settle[i], p1, p2, distance);
        system.setConstraintParameters(settle[i], p1, p2, 1.05*distance);
    }
    for (int i = 0; i < (int) shake.size(); i++) {
        int p1, p2;
        double distance;
        system.getConstraintParameters(shake[i], p1, p2, distance);
        system.setConstraintParameters(shake[i], p1, p2, 0.95*distance);
    }
    for (int i = 0; i < (int) ccma.size(); i++) {
        int p1, p2;
        double distance;
        system.getConstraintParameters(ccma[i], p1, p2, distance);
        system.setConstraintParameters(ccma[i], p1, p2, distance+0.002*(i+1));
    }
    context.updateConstraintsInContext();
    verifyConstraints(system, context, 1e-4);

    // Running dynamics must keep satisfying the new distances.

    integrator.step(20);
    verifyConstraints(system, context, 1e-4);
}

void testUpdateConstraintsPreservesState() {
    // The update must not disturb the positions or velocities.

    System system;
    vector<Vec3> positions;
    vector<int> settle, shake, ccma;
    buildConstrainedSystem(system, positions, settle, shake, ccma);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(300.0, 5);
    integrator.step(10);
    State before = context.getState(State::Positions | State::Velocities);

    int p1, p2;
    double distance;
    system.getConstraintParameters(ccma[0], p1, p2, distance);
    system.setConstraintParameters(ccma[0], p1, p2, distance+0.001);
    context.updateConstraintsInContext();

    State after = context.getState(State::Positions | State::Velocities);
    for (int i = 0; i < system.getNumParticles(); i++) {
        ASSERT_EQUAL_VEC(before.getPositions()[i], after.getPositions()[i], 1e-6);
        ASSERT_EQUAL_VEC(before.getVelocities()[i], after.getVelocities()[i], 1e-6);
    }
}

void testUpdateConstraintsRequiringRebuild() {
    // Changing only one of a SHAKE cluster's peripheral distances means it can no longer
    // be treated as a cluster, so the division between the algorithms changes.  The update
    // must fall back to rebuilding, and must still give the right answer.

    System system;
    vector<Vec3> positions;
    vector<int> settle, shake, ccma;
    buildConstrainedSystem(system, positions, settle, shake, ccma);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(300.0, 5);
    integrator.step(10);
    State before = context.getState(State::Positions | State::Velocities);

    int p1, p2;
    double distance;
    system.getConstraintParameters(shake[0], p1, p2, distance);
    system.setConstraintParameters(shake[0], p1, p2, 1.2*distance);
    context.updateConstraintsInContext();
    verifyConstraints(system, context, 1e-4);

    // Reinitializing preserves the state, so the positions and velocities should still
    // be the ones we started from.

    State after = context.getState(State::Positions | State::Velocities);
    for (int i = 0; i < system.getNumParticles(); i++)
        ASSERT_EQUAL_VEC(before.getVelocities()[i], after.getVelocities()[i], 1e-5);

    integrator.step(20);
    verifyConstraints(system, context, 1e-4);
}

void testUpdateConstraintsMatchesNewContext() {
    // A context whose constraints have been updated should behave the same as one built
    // from scratch with the new distances.

    System system1, system2;
    vector<Vec3> positions;
    vector<int> settle, shake, ccma;
    buildConstrainedSystem(system1, positions, settle, shake, ccma);
    vector<Vec3> positions2;
    vector<int> settle2, shake2, ccma2;
    buildConstrainedSystem(system2, positions2, settle2, shake2, ccma2);
    for (int i = 0; i < (int) ccma.size(); i++) {
        int p1, p2;
        double distance;
        system2.getConstraintParameters(ccma2[i], p1, p2, distance);
        system2.setConstraintParameters(ccma2[i], p1, p2, distance+0.003);
    }

    VerletIntegrator integrator1(0.001), integrator2(0.001);
    Context context1(system1, integrator1, platform);
    Context context2(system2, integrator2, platform);
    context1.setPositions(positions);
    context2.setPositions(positions);
    context1.setVelocitiesToTemperature(300.0, 7);
    context2.setVelocitiesToTemperature(300.0, 7);

    for (int i = 0; i < (int) ccma.size(); i++) {
        int p1, p2;
        double distance;
        system1.getConstraintParameters(ccma[i], p1, p2, distance);
        system1.setConstraintParameters(ccma[i], p1, p2, distance+0.003);
    }
    context1.updateConstraintsInContext();

    integrator1.step(20);
    integrator2.step(20);
    State state1 = context1.getState(State::Positions);
    State state2 = context2.getState(State::Positions);
    for (int i = 0; i < system1.getNumParticles(); i++)
        ASSERT_EQUAL_VEC(state2.getPositions()[i], state1.getPositions()[i], 1e-4);
}

void testChangingParticles() {
    // Changing which particles a constraint connects cannot be applied in place, so it
    // has to fall back to rebuilding.  The result must still be correct.

    System system;
    vector<Vec3> positions;
    vector<int> settle, shake, ccma;
    buildConstrainedSystem(system, positions, settle, shake, ccma);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    verifyConstraints(system, context, 1e-4);

    // Point the last constraint of a chain at the chain's first atom instead.

    int p1, p2;
    double distance;
    system.getConstraintParameters(ccma[2], p1, p2, distance);
    int q1, q2;
    double d2;
    system.getConstraintParameters(ccma[0], q1, q2, d2);
    system.setConstraintParameters(ccma[2], q1, p2, distance);
    context.updateConstraintsInContext();
    verifyConstraints(system, context, 1e-4);

    integrator.step(20);
    verifyConstraints(system, context, 1e-4);
}

void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testUpdateConstraints();
        testUpdateConstraintsPreservesState();
        testUpdateConstraintsRequiringRebuild();
        testUpdateConstraintsMatchesNewContext();
        testChangingParticles();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
