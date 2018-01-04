#ifndef CUDA_DRUDENOSE_KERNELS_H_
#define CUDA_DRUDENOSE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013-2015 Stanford University and the Authors.      *
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
#include "CudaContext.h"
#include "CudaArray.h"

namespace OpenMM {


/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to take one time step
 */
class CudaIntegrateDrudeNoseHooverStepKernel : public IntegrateDrudeNoseHooverStepKernel {
public:
    CudaIntegrateDrudeNoseHooverStepKernel(std::string name, const Platform& platform, CudaContext& cu) :
            IntegrateDrudeNoseHooverStepKernel(name, platform), cu(cu), normalParticles(NULL), pairParticles(NULL), kineticEnergies(NULL), normalKEBuffer(NULL), realKEBuffer(NULL), drudeKEBuffer(NULL) {
    }
    ~CudaIntegrateDrudeNoseHooverStepKernel();
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
    std::tuple<double, double, double> propagateNHChain(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator);
    CudaContext& cu;
    double prevStepSize, centerkbT, centerNkbT;
    double2 noseHooverkbT, noseHooverNkbT;
    int centerDof;
    int2 noseHooverDof;
    CudaArray* normalParticles;
    CudaArray* pairParticles;
    CudaArray* pairIsCenterParticle;
    CudaArray* kineticEnergies;
    CudaArray* normalKEBuffer;
    CudaArray* realKEBuffer;
    CudaArray* drudeKEBuffer;
    std::vector<double2> etaMass;
    std::vector<double2> eta;
    std::vector<double2> etaDot;
    std::vector<double2> etaDotDot;
    std::vector<int>    centerParticles;
    std::vector<double> etaCenterMass;
    std::vector<double> etaCenter;
    std::vector<double> etaCenterDot;
    std::vector<double> etaCenterDotDot;
    CUfunction kernelVel, kernelPos, hardwallKernel, kernelKE, kernelKESum, kernelChain;
};

} // namespace OpenMM

#endif /*CUDA_DRUDENOSE_KERNELS_H_*/
