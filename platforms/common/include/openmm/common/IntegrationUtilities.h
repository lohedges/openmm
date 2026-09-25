#ifndef OPENMM_INTEGRATIONUTILITIES_H_
#define OPENMM_INTEGRATIONUTILITIES_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2022 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeKernel.h"
#include "openmm/common/ComputeVectorTypes.h"
#include "openmm/System.h"
#include <iosfwd>
#include <map>

namespace OpenMM {

class ComputeContext;

/**
 * This class implements features that are used by many different integrators, including
 * common workspace arrays, random number generation, and enforcing constraints.
 */

class OPENMM_EXPORT_COMMON IntegrationUtilities {
public:
    IntegrationUtilities(ComputeContext& context, const System& system);
    virtual ~IntegrationUtilities() {
    }
    /**
     * Get the array which contains position deltas.  These are the amounts by
     * which the position of each atom will change in the current step.  The actual
     * positions should not be modified until after constraints have been applied.
     */
    virtual ArrayInterface& getPosDelta() = 0;
    /**
     * Get the array which contains random values.  Each element is a float4 whose components
     * are independent, normally distributed random numbers with mean 0 and variance 1.
     * Be sure to call initRandomNumberGenerator() and prepareRandomNumbers() before
     * accessing this array.
     */
    virtual ArrayInterface& getRandom() = 0;
    /**
     * Get the array which contains the current step size.
     */
    virtual ArrayInterface& getStepSize() = 0;
    /**
     * Set the size to use for the next step.
     */
    void setNextStepSize(double size);
    /**
     * Get the size that was used for the last step.
     */
    double getLastStepSize();
    /**
     * Apply constraints to the atom positions.  When calling this method, the
     * context's array of positions should contain the positions at the start of the
     * step, and the array returned by getPosDelta() should contain the intended
     * change to each position.  This method modifies the position deltas so that,
     * once they are added to the positions, constraints will be satisfied.
     *
     * @param tol             the constraint tolerance
     */
    void applyConstraints(double tol);
    /**
     * Apply constraints to the atom velocities.
     *
     * @param tol             the constraint tolerance
     */
    void applyVelocityConstraints(double tol);
    /**
     * Initialize the random number generator.  This should be called once when the
     * context is first created.  Subsequent calls will be ignored if the random
     * seed is the same as on the first call, or throw an exception if the random
     * seed is different.
     */
    void initRandomNumberGenerator(unsigned int randomNumberSeed);
    /**
     * Ensure that sufficient random numbers are available in the array, and generate new ones if not.
     *
     * @param numValues     the number of random float4's that will be required
     * @return the index in the array at which to start reading
     */
    int prepareRandomNumbers(int numValues);
    /**
     * Compute the positions of virtual sites.
     */
    void computeVirtualSites();
    /**
     * Distribute forces from virtual sites to the atoms they are based on.
     */
    virtual void distributeForcesFromVirtualSites() = 0;
    /**
     * Create a checkpoint recording the current state of the random number generator.
     * 
     * @param stream    an output stream the checkpoint data should be written to
     */
    void createCheckpoint(std::ostream& stream);
    /**
     * Load a checkpoint that was written by createCheckpoint().
     * 
     * @param stream    an input stream the checkpoint data should be read from
     */
    void loadCheckpoint(std::istream& stream);
    /**
     * Compute the kinetic energy of the system, possibly shifting the velocities in time to account
     * for a leapfrog integrator.
     * 
     * @param timeShift   the amount by which to shift the velocities in time
     */
    double computeKineticEnergy(double timeShift);
    /**
     * Compute the current velocities, shifting them in time to account for a leapfrog integrator.
     * 
     * @param timeShift   the amount by which to shift the velocities in time
     * @param velocities  the shifted velocities are returned in this
     */
    void computeShiftedVelocities(double timeShift, std::vector<Vec3>& velocities);
    /**
     * Update the constraint parameters to match the ones currently stored in the System.
     * Only the distances may have changed.  The particles involved in each constraint must
     * be the same as when the Context was created.
     *
     * This does not rebuild the division of constraints between the SETTLE, SHAKE and CCMA
     * algorithms.  If the new distances would change that division, it cannot be applied
     * this way and the caller must reinitialize the Context instead.
     *
     * @param system   the System whose constraint parameters should be copied
     * @return true if the parameters were updated, false if the Context must be reinitialized
     */
    bool updateConstraints(const System& system);
protected:
    virtual void applyConstraintsImpl(bool constrainVelocities, double tol) = 0;
    /**
     * Records where a constraint ended up when the constraints were divided between the
     * SETTLE, SHAKE and CCMA algorithms, and which particles it connects, so that its
     * distance can be updated later.
     *
     * IGNORED marks a constraint between two massless particles, which is left out of all
     * three algorithms.  NONE means no algorithm was recorded for it, which should not
     * happen and is treated as a reason to rebuild rather than update in place.
     */
    struct ConstraintLocation {
        enum Algorithm {NONE, IGNORED, SETTLE, SHAKE, CCMA};
        Algorithm algorithm;
        int index;
        int particle1, particle2;
        ConstraintLocation() : algorithm(NONE), index(-1), particle1(-1), particle2(-1) {
        }
    };
    ComputeContext& context;
    ComputeKernel settlePosKernel, settleVelKernel;
    ComputeKernel shakePosKernel, shakeVelKernel;
    ComputeKernel ccmaDirectionsKernel, ccmaPosForceKernel, ccmaVelForceKernel;
    ComputeKernel ccmaMultiplyKernel, ccmaUpdateKernel, ccmaFullKernel;
    ComputeKernel vsitePositionKernel, vsiteForceKernel, vsiteSaveForcesKernel;
    ComputeKernel randomKernel, timeShiftKernel, kineticEnergyKernel;
    ComputeArray posDelta;
    ComputeArray settleAtoms;
    ComputeArray settleParams;
    ComputeArray shakeAtoms;
    ComputeArray shakeParams;
    ComputeArray random;
    ComputeArray randomSeed;
    ComputeArray stepSize;
    ComputeArray ccmaAtoms;
    ComputeArray ccmaConstraintAtoms;
    ComputeArray ccmaDistance;
    ComputeArray ccmaReducedMass;
    ComputeArray ccmaAtomConstraints;
    ComputeArray ccmaNumAtomConstraints;
    ComputeArray ccmaConstraintMatrixColumn;
    ComputeArray ccmaConstraintMatrixValue;
    ComputeArray ccmaDelta1;
    ComputeArray ccmaDelta2;
    ComputeArray ccmaConverged;
    ComputeArray vsite2AvgAtoms, vsite2AvgWeights;
    ComputeArray vsite3AvgAtoms, vsite3AvgWeights;
    ComputeArray vsiteOutOfPlaneAtoms, vsiteOutOfPlaneWeights;
    ComputeArray vsiteLocalCoordsIndex, vsiteLocalCoordsAtoms, vsiteLocalCoordsWeights;
    ComputeArray vsiteLocalCoordsPos, vsiteLocalCoordsStartIndex;
    ComputeArray vsiteSymmetryAtoms, vsiteSymmetryMatrix, vsiteSymmetryOffset, vsiteSymmetryUseBox;
    ComputeArray vsiteStage;
    ComputeArray kineticEnergy;
    int randomPos, lastSeed, numVsites, numVsiteStages, keWorkGroupSize;
    // State recorded when the constraints were set up, so that their distances can be
    // updated later without rebuilding everything.  All of these are indexed by the
    // constraint's index in the System, except settleClusterConstraints and
    // shakeClusterConstraints which are indexed by cluster.
    std::vector<double> constraintDistance;
    std::vector<ConstraintLocation> constraintLocation;
    std::vector<std::vector<int> > settleClusterConstraints;
    std::vector<std::vector<int> > shakeClusterConstraints;
    std::vector<mm_float4> shakeParamsVec;
    bool hasOverlappingVsites;
    mm_double2 lastStepSize;
    struct ShakeCluster;
    struct ConstraintOrderer;
};

} // namespace OpenMM

#endif /*OPENMM_INTEGRATIONUTILITIES_H_*/
