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
            IntegrateDrudeNoseHooverStepKernel(name, platform), cu(cu), normalParticles(NULL), pairParticles(NULL), vscaleFactors(NULL), particleResId(NULL), particlesInResidues(NULL), comVelm(NULL), normVelm(NULL), kineticEnergies(NULL) {
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
     * @param context       the context in which to execute this kernel
     * @param integrator    the DrudeNoseHooverIntegrator this kernel is being used for
     * @param isKESumValid  whether use saved KESum or calculate based on new velocities
     */
    double computeKineticEnergy(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator, bool isKESumValid);
private:
    std::vector<double> propagateNHChain(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator);
    void assignVscaleFactors();
    CudaContext& cu;
    double prevStepSize, drudekbT, drudeNkbT, realkbT, KESum;
    int drudeDof, numResidues, numParticles;
    CudaArray* normalParticles;
    CudaArray* pairParticles;
    CudaArray* vscaleFactors;
    CudaArray* particleResId;
    CudaArray* particleTempGroup;
    CudaArray* particlesInResidues;
    CudaArray* comVelm;
    CudaArray* normVelm;
    CudaArray* kineticEnergyBuffer;
    CudaArray* kineticEnergies;
    std::vector<std::vector<double> > etaMass;
    std::vector<std::vector<double> > eta;
    std::vector<std::vector<double> > etaDot;
    std::vector<std::vector<double> > etaDotDot;
    std::vector<int>    tempGroupDof;
    std::vector<int>    particleResIdVec;
    std::vector<int2>   particlesInResiduesVec;
    std::vector<int>    particleTempGroupVec;
    std::vector<int>    normalParticleVec;
    std::vector<int2>   pairParticleVec;
    std::vector<double>    tempGroupNkbT;
    std::vector<double>    tempGroupVscaleFactors;
    std::vector<double>    vscaleFactorsVec;
    CUfunction kernelVel, kernelPos, hardwallKernel, kernelKE, kernelKESum, kernelChain, kernelNormVel, kernelCOMVel;
};

} // namespace OpenMM

#endif /*CUDA_DRUDENOSE_KERNELS_H_*/
