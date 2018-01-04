#ifndef REFERENCE_DRUDENOSE_KERNELS_H_
#define REFERENCE_DRUDENOSE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013 Stanford University and the Authors.           *
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

#include "openmm/DrudeNoseKernels.h"
#include "ReferencePlatform.h"
#include "RealVec.h"
#include <utility>
#include <vector>

class ReferenceConstraintAlgorithm;

namespace OpenMM {


/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to take one time step
 */
class ReferenceIntegrateDrudeNoseHooverStepKernel : public IntegrateDrudeNoseHooverStepKernel {
public:
    ReferenceIntegrateDrudeNoseHooverStepKernel(std::string name, const Platform& platform, ReferencePlatform::PlatformData& data) :
        IntegrateDrudeNoseHooverStepKernel(name, platform), data(data) {
    }
    ~ReferenceIntegrateDrudeNoseHooverStepKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
     * @param force      the DrudeForce to get particle parameters from
     */
    void initialize(const System& system, const DrudeNoseHooverIntegrator& integrator, const DrudeForce& force);
    /**
     * Execute the kernel.
     *
     * @param context        the context in which to execute this kernel
     * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
     */
    void execute(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator);
    /**
     * Compute the kinetic energy.
     * 
     * @param context     the context in which to execute this kernel
     * @param integrator  the DrudeNoseHooverIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator);
private:
    /**
     * Compute the kinetic energies for each degrees of freedom
     * 
     * @param context     the context in which to execute this kernel
     */
    //void computeNoseKineticEnergy(ContextImpl& context);
    /**
     * Perform half-step update of dual Nose-Hoover chain thermostat.
     * 
     * @param context     the context in which to execute this kernel
     * @param integrator  the DrudeNoseHooverIntegrator this kernel is being used for
     */
    void propagateNHChain(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator);
    /**
     * Perform half-step update of velocity Verlet move.
     * 
     * @param context     the context in which to execute this kernel
     * @param integrator  the DrudeNoseHooverIntegrator this kernel is being used for
     */
    void propagateHalfVelocity(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator);
    ReferencePlatform::PlatformData& data;
    std::vector<int> normalParticles;
    std::vector<int> centerParticles;
    std::vector<std::pair<int, int> > pairParticles;
    std::vector<bool> pairIsCenterParticle;
    std::vector<double> particleMass;
    std::vector<double> particleInvMass;
    std::vector<double> pairInvTotalMass;
    std::vector<double> pairInvReducedMass;
    std::vector<double> centerInvTotalMass;
    std::vector<double> etaMass;
    std::vector<double> eta;
    std::vector<double> etaDot;
    std::vector<double> etaDotDot;
    double realkbT, drudekbT, realNkbT, drudeNkbT, centerkbT, centerNkbT;
    int realDof, drudeDof, centerDof, numTempGroup, idxMaxNHChains, iNumNHChains;
};


} // namespace OpenMM

#endif /*REFERENCE_DRUDENOSE_KERNELS_H_*/
