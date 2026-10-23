/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2024 Stanford University and the Authors.      *
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

#include "CudaTests.h"
#include "TestNonbondedForce.h"
#include "CudaContext.h"
#include "openmm/internal/ContextImpl.h"
#include <cuda.h>
#include <string>

void testDeterministicForces() {
    // Check that the CudaDeterministicForces property works correctly.
    
    const int numParticles = 1000;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(6, 0, 0), Vec3(2.1, 6, 0), Vec3(-1.5, -0.5, 6));
    NonbondedForce *nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    system.addForce(nonbonded);
    vector<Vec3> positions;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(i%2 == 0 ? 1 : -1, 1, 0);
        positions.push_back(Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5)*6);
    }
    VerletIntegrator integrator(0.001);
    map<string, string> properties;
    properties[CudaPlatform::CudaDeterministicForces()] = "true";
    Context context(system, integrator, platform, properties);
    context.setPositions(positions);
    State state1 = context.getState(State::Forces);
    State state2 = context.getState(State::Forces);
    
    // All forces should be *exactly* equal.
    
    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL(state1.getForces()[i][0], state2.getForces()[i][0]);
        ASSERT_EQUAL(state1.getForces()[i][1], state2.getForces()[i][1]);
        ASSERT_EQUAL(state1.getForces()[i][2], state2.getForces()[i][2]);
    }
}

/**
 * This is used to get access to the CudaContext underlying a Context.
 */
class CudaContextAccessor : public HarmonicBondForce {
public:
    CudaContext& getCudaContext(Context& context) const {
        CudaPlatform::PlatformData* data = reinterpret_cast<CudaPlatform::PlatformData*>(getContextImpl(context).getPlatformData());
        return *data->contexts[0];
    }
};

void testReorderAfterLargePositionChange() {
    // Setting positions that are very different from the current ones should make the atoms get reordered
    // once the context is stepped, without waiting for the usual minimum number of steps.  Single point
    // evaluations and small changes should not trigger it.

    const int numMolecules = 3000;
    const double boxSize = 6.0;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    nonbonded->setCutoffDistance(1.0);
    for (int i = 0; i < numMolecules; i++) {
        system.addParticle(1.0);
        system.addParticle(1.0);
        nonbonded->addParticle(0.5, 0.2, 0.5);
        nonbonded->addParticle(-0.5, 0.2, 0.5);
        nonbonded->addException(2*i, 2*i+1, 0.0, 1.0, 0.0);
    }
    system.addForce(nonbonded);

    // Build two unrelated configurations by putting the molecules on randomly chosen sites of a grid.

    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    const int gridSize = 15;
    vector<vector<Vec3> > configs(2);
    for (int c = 0; c < 2; c++) {
        vector<int> sites(gridSize*gridSize*gridSize);
        for (int i = 0; i < (int) sites.size(); i++)
            sites[i] = i;
        for (int i = sites.size()-1; i > 0; i--)
            swap(sites[i], sites[(int) (genrand_real2(sfmt)*(i+1))]);
        for (int i = 0; i < numMolecules; i++) {
            int site = sites[i];
            Vec3 center((site%gridSize + 0.5)*boxSize/gridSize, ((site/gridSize)%gridSize + 0.5)*boxSize/gridSize, (site/(gridSize*gridSize) + 0.5)*boxSize/gridSize);
            center += Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5)*0.1;
            configs[c].push_back(center);
            configs[c].push_back(center+Vec3(0.15, 0, 0));
        }
    }
    VerletIntegrator integrator(0.0001);
    Context context(system, integrator, platform);
    CudaContext& cu = CudaContextAccessor().getCudaContext(context);
    context.setPositions(configs[0]);
    integrator.step(10);

    // Evaluate the energy of each configuration in turn without stepping.  That should not trigger a reorder.

    context.setPositions(configs[1]);
    context.getState(State::Energy);
    context.setPositions(configs[0]);
    context.getState(State::Energy);
    ASSERT(cu.getStepsSinceReorder() > 0);

    // Load the other configuration and step.  Few enough steps have been taken that only the check on setting
    // positions can trigger a reorder, which happens after the second step.

    context.setPositions(configs[1]);
    context.getState(State::Energy);
    integrator.step(2);
    ASSERT_EQUAL(0, cu.getStepsSinceReorder());

    // Move a single molecule.  That should not trigger a reorder.

    vector<Vec3> positions = configs[1];
    positions[20] += Vec3(0.1, -0.05, 0.07);
    positions[21] += Vec3(0.1, -0.05, 0.07);
    context.setPositions(positions);
    context.getState(State::Energy);
    integrator.step(2);
    ASSERT(cu.getStepsSinceReorder() > 0);
}

bool canRunHugeTest() {
    // Create a minimal context just to see which device is being used.

    System system;
    system.addParticle(1.0);
    VerletIntegrator integrator(1.0);
    Context context(system, integrator, platform);
    int deviceIndex = stoi(platform.getPropertyValue(context, CudaPlatform::CudaDeviceIndex()));

    // Find out how much memory the device has.

    CUdevice device;
    cuDeviceGet(&device, deviceIndex);
    size_t memory;
    cuDeviceTotalMem(&memory, device);

    // Only run the huge test if the device has at least 4 GB of memory.

    return (memory >= 4L*(1<<30));
}

void runPlatformTests() {
    testParallelComputation(NonbondedForce::NoCutoff);
    testParallelComputation(NonbondedForce::Ewald);
    testParallelComputation(NonbondedForce::PME);
    testParallelComputation(NonbondedForce::LJPME);
    testReordering();
    testDeterministicForces();
    testReorderAfterLargePositionChange();
    if (canRunHugeTest())
        testHugeSystem();
}
