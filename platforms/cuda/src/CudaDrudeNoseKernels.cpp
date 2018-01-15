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

#include "CudaDrudeNoseKernels.h"
#include "CudaDrudeNoseKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/CMMotionRemover.h"
#include "CudaBondedUtilities.h"
#include "CudaForceInfo.h"
#include "CudaIntegrationUtilities.h"
#include "SimTKOpenMMRealType.h"
#include <algorithm>
#include <set>
#include <typeinfo>
#include <iostream>
#include <sys/time.h>



using namespace OpenMM;
using namespace std;


CudaIntegrateDrudeNoseHooverStepKernel::~CudaIntegrateDrudeNoseHooverStepKernel() {
    if (normalParticles != NULL)
        delete normalParticles;
    if (pairParticles != NULL)
        delete pairParticles;
    if (vscaleFactors != NULL)
        delete vscaleFactors;
    if (particleResId != NULL)
        delete particleResId;
    if (particleTempGroup != NULL)
        delete particleTempGroup;
    if (particlesInResidues != NULL)
        delete particlesInResidues;
    if (comVelm != NULL)
        delete comVelm;
    if (normVelm != NULL)
        delete normVelm;
    if (kineticEnergies != NULL)
        delete kineticEnergies;
}

void CudaIntegrateDrudeNoseHooverStepKernel::initialize(const System& system, const DrudeNoseHooverIntegrator& integrator, const DrudeForce& force) {
    cu.getPlatformData().initializeContexts(system);

    drudeDof = 0;
    drudekbT = BOLTZ * integrator.getDrudeTemperature();
    realkbT = BOLTZ * integrator.getTemperature();

    int numDrudeSteps = integrator.getDrudeStepsPerRealStep();
    int numNHChains = integrator.getNumNHChains();

    // initialize residue masses to zero
    int numResidues = integrator.getNumResidues();
    for (int i=0; i < numResidues; i++)
        particlesInResiduesVec.push_back(make_int2(0,-1));


    // initialize eta values to zero
    int numTempGroups = integrator.getNumTempGroups();
    etaMass = std::vector<vector<double> >(numTempGroups+2, std::vector<double>(numNHChains, 0.0));
    eta = std::vector<vector<double> >(numTempGroups+2, std::vector<double>(numNHChains, 0.0));
    etaDot = std::vector<vector<double> >(numTempGroups+2, std::vector<double>(numNHChains+1, 0.0));
    etaDotDot = std::vector<vector<double> >(numTempGroups+2, std::vector<double>(numNHChains, 0.0));
    std::vector<double> tempGroupRedMass(numTempGroups+1,0.0);

    /*eta.resize(numTempGroups+1, std::vector<double>(numNHChains));
    etaDot.resize(numTempGroups+1, std::vector<double>(numNHChains+1));
    etaDotDot.resize(numTempGroups+1, std::vector<double>(numNHChains)); */
    for (int i=0; i < numTempGroups+2; i++) {
        tempGroupDof.push_back(0);
        vscaleFactorsVec.push_back(1.0);
    }


    // Identify particle pairs and ordinary particles.
    
    set<int> particles;
    int prevResId = -1;

    for (int i = 0; i < system.getNumParticles(); i++) {
        int tg;
        particles.insert(i);
        integrator.getParticleTempGroup(i, tg);
        particleTempGroupVec.push_back(tg);
        int resid = integrator.getParticleResId(i);
        particleResIdVec.push_back(resid);
        particlesInResiduesVec[resid].x += 1;
        if (prevResId != resid) {
            particlesInResiduesVec[resid].y = i;
            prevResId = resid;
        }
        double mass = system.getParticleMass(i);
        double resInvMass = integrator.getResInvMass(resid);
        if (mass != 0.0) {
            tempGroupDof[tg] += 3;
            tempGroupRedMass[tg] += 3 * mass * resInvMass;
        }
    }
    for (int i = 0; i < force.getNumParticles(); i++) {
        int p, p1, p2, p3, p4;
        int tg, tg1;
        double charge, polarizability, aniso12, aniso34;
        double m, m1;
        force.getParticleParameters(i, p, p1, p2, p3, p4, charge, polarizability, aniso12, aniso34);
        particles.erase(p);
        particles.erase(p1);
        integrator.getParticleTempGroup(p, tg);
        integrator.getParticleTempGroup(p1, tg1);
        if (tg != tg1)
            throw OpenMMException("Temperature group for drude particle must be the same as the parent particle");
        pairParticleVec.push_back(make_int2(p, p1));
        tempGroupDof[tg] -= 3;
        drudeDof += 3;
    }
    normalParticleVec.insert(normalParticleVec.begin(), particles.begin(), particles.end());
    normalParticles = CudaArray::create<int>(cu, max((int) normalParticleVec.size(), 1), "drudeNormalParticles");
    pairParticles = CudaArray::create<int2>(cu, max((int) pairParticleVec.size(), 1), "drudePairParticles");
    particleResId = CudaArray::create<int>(cu, max((int) particleResIdVec.size(), 1), "drudeParticleResId");
    particleTempGroup = CudaArray::create<int>(cu, max((int) particleTempGroupVec.size(), 1), "drudeParticleTempGroups");
    particlesInResidues = CudaArray::create<int2>(cu, max((int) particlesInResiduesVec.size(), 1), "drudeParticlesInResidues");

    int numAtoms = cu.getNumAtoms();
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        vscaleFactors = CudaArray::create<double>(cu, max((int) vscaleFactorsVec.size(), 1), "drudeScaleFactors");
        comVelm = CudaArray::create<double4>(cu, max(numResidues, 1), "drudeComVelm");
        normVelm = CudaArray::create<double4>(cu, max(numAtoms, 1), "drudeNormVelm");
    }
    else {
        vscaleFactors = CudaArray::create<float>(cu, max((int) vscaleFactorsVec.size(), 1), "drudeScaleFactors");
        comVelm = CudaArray::create<float4>(cu, max(numResidues, 1), "drudeComVelm");
        normVelm = CudaArray::create<float4>(cu, max(numAtoms, 1), "drudeNormVelm");
    }
        
    if (normalParticleVec.size() > 0)
        normalParticles->upload(normalParticleVec);
    if (pairParticleVec.size() > 0)
        pairParticles->upload(pairParticleVec);
    if (particleResIdVec.size() > 0)
        particleResId->upload(particleResIdVec);
    if (particleTempGroupVec.size() > 0)
        particleTempGroup->upload(particleTempGroupVec);
    if (particlesInResiduesVec.size() > 0)
        particlesInResidues->upload(particlesInResiduesVec);

    // reduce real d.o.f by number of constraints, and 3 if CMMotion remove is true
    for (int i = 0; i < system.getNumConstraints(); i++) {
        int p,p1,tg,tg1;
        double distance;
        system.getConstraintParameters(i, p, p1, distance);
        integrator.getParticleTempGroup(p, tg);
        integrator.getParticleTempGroup(p1, tg1);
        if (tg != tg1)
            throw OpenMMException("Temperature group of constrained particles must be the same");

        tempGroupDof[tg] -= 1;
    }
    tempGroupDof[numTempGroups] = 3*integrator.getNumResidues();
    tempGroupDof[numTempGroups+1] = drudeDof;

    /* Ignore CM motion removal, which is small relative to the large d.o.f. for condensed phase
    for (int i = 0; i < system.getNumForces(); i++) {
        cout << typeid(system.getForce(i)).name() << "\n";
        if (typeid(system.getForce(i)) == typeid(CMMotionRemover)) {
            cout << "CMMotion removal found, reduce dof by 3\n";
            noseHooverDof.x -= 3;
            break;
        }
    }
    */

    // calculate etaMass
    drudeNkbT = drudeDof * drudekbT;
    double drudeEtaMassUnit = drudekbT * pow(integrator.getDrudeCouplingTime(), 2);  // internal
    double realEtaMassUnit = realkbT * pow(integrator.getCouplingTime(), 2);
    for (int i=0; i < numTempGroups+1; i++) {
        tempGroupNkbT.push_back((tempGroupDof[i] - tempGroupRedMass[i]) * realkbT);
        etaMass[i][0] = (tempGroupDof[i] - tempGroupRedMass[i]) * realEtaMassUnit;
        for (int ich=1; ich < integrator.getNumNHChains(); ich++) {
            etaMass[i][ich] = realEtaMassUnit;
            etaDotDot[i][ich] = (etaMass[i][ich-1] * etaDot[i][ich-1] * etaDot[i][ich-1] - realkbT) / etaMass[i][ich];
        }
    }
    // drude temp group
    int itg = numTempGroups+1;
    tempGroupNkbT.push_back(drudeNkbT);
    etaMass[itg][0] = drudeDof * drudeEtaMassUnit;
    for (int ich=1; ich < integrator.getNumNHChains(); ich++) {
        etaMass[itg][ich] = drudeEtaMassUnit;
        if (integrator.getUseDrudeNHChains()) {
            etaDotDot[itg][ich] = (etaMass[itg][ich-1] * etaDot[itg][ich-1] * etaDot[itg][ich-1] - drudekbT) / etaMass[itg][ich];
        }
    }


    cout << "Initialization finished\n";
    cout << "real T : " << integrator.getTemperature() << ", drude T : " << integrator.getDrudeTemperature() << "\n";
    for (int i=0; i< numTempGroups+1; i++) {
        cout << "real NkbT[" << i << "] : " << tempGroupNkbT[i] << ", etaMass[" << i << "] : " << etaMass[i][0] << "real Dof[" << i << "] : " << tempGroupDof[i] - tempGroupRedMass[i] << "\n";
    }
    cout << "drude NkbT : " << drudeNkbT << "drudeQ0 : " << etaMass[itg][0] << ", drude Dof : " << tempGroupDof[itg] << "\n";
    cout << "Num NH Chain : " << integrator.getNumNHChains() << "\n";
    cout << "Use NH Chain for Drude dof : " << integrator.getUseDrudeNHChains() << "\n";
    cout << "real couplingTime : " << integrator.getCouplingTime() << "\n";
    cout << "drude couplingTime : " << integrator.getDrudeCouplingTime() << "\n";
    cout << "pair Particles[0].x : " << pairParticleVec[0].x << "\n";
    cout << "pair Particles[0].y : " << pairParticleVec[0].y << "\n";

    // extra dummy chain which will always have etaDot = 0

    // Create kernels.
    int elementSize = (cu.getUseDoublePrecision() || cu.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
    cout << "ElementSize : " << elementSize << ", double : " << sizeof(double) << ", float : " << sizeof(float) << "\n";
    kineticEnergies = new CudaArray(cu, 2, elementSize, "kineticEnergies");
    
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_NORMAL_PARTICLES"] = cu.intToString(normalParticleVec.size());
    defines["NUM_RESIDUES"] = cu.intToString(numResidues);
    defines["NUM_TEMP_GROUPS"] = cu.intToString(numTempGroups);
    defines["NUM_PAIRS"] = cu.intToString(pairParticleVec.size());
    defines["WORK_GROUP_SIZE"] = cu.intToString(CudaContext::ThreadBlockSize);
    cout << "NUM_ATOMS : " << cu.getNumAtoms() << ", PADDED_NUM_ATOMS : " << cu.getPaddedNumAtoms() << ", NUM_NORMAL_PARTICLES : " << normalParticleVec.size() << ", NUM_PAIRS : " << pairParticleVec.size() << ", WORK_GROUP_SIZE : " << CudaContext::ThreadBlockSize << "\n";
    //defines["NUM_DRUDE_STEPS"] = cu.intToString(numDrudeSteps);
    map<string, string> replacements;
    CUmodule module = cu.createModule(CudaDrudeNoseKernelSources::vectorOps+CudaDrudeNoseKernelSources::drudeNoseHoover, defines, "");
    kernelKE = cu.getKernel(module, "computeDrudeNoseHooverKineticEnergies");
    kernelKESum = cu.getKernel(module, "sumDrudeKineticEnergies");
    kernelCOMVel = cu.getKernel(module, "calcCOMVelocities");
    kernelNormVel = cu.getKernel(module, "normalizeVelocities");
    kernelChain = cu.getKernel(module, "integrateDrudeNoseHooverChain");
    kernelVel = cu.getKernel(module, "integrateDrudeNoseHooverVelocities");
    kernelPos = cu.getKernel(module, "integrateDrudeNoseHooverPositions");
    hardwallKernel = cu.getKernel(module, "applyHardWallConstraints");
    prevStepSize = -1.0;
    cout << "Cuda Modules are created\n";
}

void CudaIntegrateDrudeNoseHooverStepKernel::execute(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities& integration = cu.getIntegrationUtilities();
    int numAtoms = cu.getNumAtoms();
    int    numTempGroups = integrator.getNumTempGroups();
    
    // Compute integrator coefficients.
    
    double stepSize = integrator.getStepSize();
    //cout << "stepSize : " << stepSize << "\n";

    double fscale = 0.5*stepSize/(double) 0x100000000;
    double vscaleDrude = 1.0;
    double fscaleDrude = fscale;
    double maxDrudeDistance = integrator.getMaxDrudeDistance();
    double hardwallscaleDrude = sqrt(BOLTZ*integrator.getDrudeTemperature());
    if (stepSize != prevStepSize) {
        if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
            double2 ss = make_double2(0, stepSize);
            integration.getStepSize().upload(&ss);
        }
        else {
            float2 ss = make_float2(0, (float) stepSize);
            integration.getStepSize().upload(&ss);
        }
        prevStepSize = stepSize;
    }
    
    // Create appropriate pointer for the precision mode.
    
    float fscaleFloat = (float) fscale;
    float vscaleDrudeFloat = (float) vscaleDrude;
    float fscaleDrudeFloat = (float) fscaleDrude;
    float maxDrudeDistanceFloat =(float) maxDrudeDistance;
    float hardwallscaleDrudeFloat = (float) hardwallscaleDrude;
    void *fscalePtr, *vscaleDrudePtr, *fscaleDrudePtr, *maxDrudeDistancePtr, *hardwallscaleDrudePtr;
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        fscalePtr = &fscale;
        vscaleDrudePtr = &vscaleDrude;
        fscaleDrudePtr = &fscaleDrude;
        maxDrudeDistancePtr = &maxDrudeDistance;
        hardwallscaleDrudePtr = &hardwallscaleDrude;
    }
    else {
        fscalePtr = &fscaleFloat;
        vscaleDrudePtr = &vscaleDrudeFloat;
        fscaleDrudePtr = &fscaleDrudeFloat;
        maxDrudeDistancePtr = &maxDrudeDistanceFloat;
        hardwallscaleDrudePtr = &hardwallscaleDrudeFloat;
    }

    // First half of velocity + thermostat integration.
    vscaleFactorsVec = propagateNHChain(context, integrator);
    vscaleFactors->upload(vscaleFactorsVec);
    //assignVscaleFactors();
    //vscaleDrude = tempGroupVscaleFactors[numTempGroups+1];
    //vscaleDrudeFloat = (float) vscaleDrude;
    //cout << "vscaleDrude at NHChain : " << vscaleDrude << "\n";

    //cout << "vscale : " << vscale << ", vscaleDrude : " << vscaleDrude << "\n";


    bool updatePosDelta = false;
    void *updatePosDeltaPtr = &updatePosDelta;
    void* argsChain[] = {&cu.getVelm().getDevicePointer(), &normVelm->getDevicePointer(),
            &normalParticles->getDevicePointer(), &pairParticles->getDevicePointer(), &particleTempGroup->getDevicePointer(), &vscaleFactors->getDevicePointer()};
    cu.executeKernel(kernelChain, argsChain, numAtoms);

    // Call the first half of velocity integration kernel. (both thermostat and actual velocity update)
    updatePosDelta = true;
    void* argsVel[] = {&cu.getVelm().getDevicePointer(), &cu.getForce().getDevicePointer(), &integration.getPosDelta().getDevicePointer(),
            &normalParticles->getDevicePointer(), &pairParticles->getDevicePointer(), &integration.getStepSize().getDevicePointer(),
            fscalePtr, fscaleDrudePtr, updatePosDeltaPtr};
    cu.executeKernel(kernelVel, argsVel, numAtoms);

    // Apply position constraints.
    integration.applyConstraints(integrator.getConstraintTolerance());

    // Call the position integration kernel.
    CUdeviceptr posCorrection = (cu.getUseMixedPrecision() ? cu.getPosqCorrection().getDevicePointer() : 0);
    void* argsPos[] = {&cu.getPosq().getDevicePointer(), &posCorrection, &integration.getPosDelta().getDevicePointer(),
            &cu.getVelm().getDevicePointer(), &integration.getStepSize().getDevicePointer()};
    cu.executeKernel(kernelPos, argsPos, numAtoms);
    
    // Apply hard wall constraints.
    if (maxDrudeDistance > 0) {
        void* hardwallArgs[] = {&cu.getPosq().getDevicePointer(), &posCorrection, &cu.getVelm().getDevicePointer(),
                &pairParticles->getDevicePointer(), &integration.getStepSize().getDevicePointer(), maxDrudeDistancePtr, hardwallscaleDrudePtr};
        cu.executeKernel(hardwallKernel, hardwallArgs, pairParticles->getSize());
    }
    integration.computeVirtualSites();

    // Calculate the force from updated position
    context.calcForcesAndEnergy(true, false);
 

    // Call the second half of velocity integration kernel. (actual velocity update only)
    updatePosDelta = false;
    void* argsVel2[] = {&cu.getVelm().getDevicePointer(), &cu.getForce().getDevicePointer(), &integration.getPosDelta().getDevicePointer(),
            &normalParticles->getDevicePointer(), &pairParticles->getDevicePointer(), &integration.getStepSize().getDevicePointer(),
            fscalePtr, fscaleDrudePtr, updatePosDeltaPtr};
    cu.executeKernel(kernelVel, argsVel2, numAtoms);

    // Apply velocity constraints
    integration.applyVelocityConstraints(integrator.getConstraintTolerance());

    // Second half of thermostat integration.
    vscaleFactorsVec = propagateNHChain(context, integrator);
    vscaleFactors->upload(vscaleFactorsVec);
    //assignVscaleFactors();
    //vscaleDrude = tempGroupVscaleFactors[numTempGroups+1];
    //vscaleDrudeFloat = (float) vscaleDrude;
    //cout << "vscaleDrude at NHChain : " << vscaleDrude << "\n";

    // Call the second half of velocity integration kernel. (Nose-Hoover chain thermostat update only)
    cu.executeKernel(kernelChain, argsChain, numAtoms);

    // Update the time and step count.
    cu.setTime(cu.getTime()+stepSize);
    cu.setStepCount(cu.getStepCount()+1);
    cu.reorderAtoms();
}

/* ----------------------------------------------------------------------
   assign and upload vscale factors for each particles to the cuda array
------------------------------------------------------------------------- */
void CudaIntegrateDrudeNoseHooverStepKernel::assignVscaleFactors() {
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        vector<double> vscaleFactorsVec;
        for (int i=0; i < tempGroupVscaleFactors.size(); i++) {
            vscaleFactorsVec.push_back(tempGroupVscaleFactors[i]);
        }
        vscaleFactors->upload(vscaleFactorsVec);
    }
    else {
        vector<float> vscaleFactorsVec;
        for (int i=0; i < tempGroupVscaleFactors.size(); i++) {
            vscaleFactorsVec.push_back((float) tempGroupVscaleFactors[i]);
        }
        vscaleFactors->upload(vscaleFactorsVec);
    }
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables
------------------------------------------------------------------------- */
std::vector<double> CudaIntegrateDrudeNoseHooverStepKernel::propagateNHChain(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator) {
    int numAtoms = cu.getNumAtoms();

    double stepSize = integrator.getStepSize();
    int    numDrudeSteps = integrator.getDrudeStepsPerRealStep();
    int    numTempGroups = integrator.getNumTempGroups();
    int    numResidues   = integrator.getNumResidues();
    double dtc = stepSize/numDrudeSteps;
    double dtc2 = dtc/2.0;
    double dtc4 = dtc/4.0;
    double dtc8 = dtc/8.0;


    vscaleFactorsVec = vector<double>(numTempGroups+2, 1.0);
    vector<double> kineticEnergiesVec(numTempGroups+2,0.0);
//    timespec t0,t1,t2,t3;
//    clock_gettime(CLOCK_REALTIME, &t0); // Works on Linux
//
//    // Compute kinetic energies of each degree of freedom.
//    cu.clearBuffer(*normalKEBuffer);
//    cu.clearBuffer(*realKEBuffer);
//    cu.clearBuffer(*drudeKEBuffer);
//    void* argsKE[] = {&cu.getVelm().getDevicePointer(),
//            &normalParticles->getDevicePointer(), &pairParticles->getDevicePointer(), &normalKEBuffer->getDevicePointer(),
//            &realKEBuffer->getDevicePointer(), &drudeKEBuffer->getDevicePointer()};
//    cu.executeKernel(kernelKE, argsKE, numAtoms);
//
//    void* argsKESum[] = {&normalKEBuffer->getDevicePointer(), &realKEBuffer->getDevicePointer(), &drudeKEBuffer->getDevicePointer(),
//            &kineticEnergies->getDevicePointer()};
//    cu.executeKernel(kernelKESum, argsKESum, CudaContext::ThreadBlockSize, CudaContext::ThreadBlockSize);
//
//    kineticEnergies->download(kineticEnergiesVec);
//    cout << "kineticEnergiesVec : " << kineticEnergiesVec[0] << ", " << kineticEnergiesVec[1] << "\n";
//    cout << "Before NHChain real T : " << kineticEnergiesVec[0]/noseHooverDof.x/BOLTZ << ", drude T : " << kineticEnergiesVec[1]/noseHooverDof.y/BOLTZ << "\n";
//    clock_gettime(CLOCK_REALTIME, &t1); // Works on Linux
//    cout << "duration : " <<  (t1.tv_nsec - t0.tv_nsec ) << '\n';

    void* argsCOMVel[] = {&cu.getVelm().getDevicePointer(),
            &particlesInResidues->getDevicePointer(), &comVelm->getDevicePointer()};

    cu.executeKernel(kernelCOMVel, argsCOMVel, numResidues);
    //cout <<  "kernelCOMVel executed ! \n";
    vector<double4> comVelmVec(numResidues);
    comVelm->download(comVelmVec);
    //comVelm->download(cu.getPinnedBuffer());
    //double4* comVelmVec = (double4*) cu.getPinnedBuffer();

    void* argsNormVel[] = {&cu.getVelm().getDevicePointer(), &particleResId->getDevicePointer(),
            &comVelm->getDevicePointer(), &normVelm->getDevicePointer()};

    cu.executeKernel(kernelNormVel, argsNormVel, numAtoms);
    //cout <<  "kernelNormVel executed ! \n";
    vector<double4> normVelmVec(numAtoms);
    normVelm->download(normVelmVec);
    //normVelm->download(cu.getPinnedBuffer());
    //double4* normVelmVec = (double4*) cu.getPinnedBuffer();


//    for (int i = 0; i < context.getSystem().getNumParticles(); i++) {
//        double m = context.getSystem().getParticleMass(i);
//        int resid = integrator.getParticleResId(i);
//        molVel[resid] += vel[i]*m;
//    }
    for (int i = 0; i < numResidues; i++) {
        Vec3 molVel = Vec3(comVelmVec[i].x,comVelmVec[i].y,comVelmVec[i].z);
        kineticEnergiesVec[numTempGroups] += (molVel.dot(molVel))/comVelmVec[i].w;
        //cout << "molvel : " << comVelmVec[i].x << " " << comVelmVec[i].y << " " << comVelmVec[i].z << "\n";
        //cout << "KECOM : " << kineticEnergiesVec[numTempGroups] << ", KECOM[" << i << "] : " << (molVel.dot(molVel))/comVelmVec[i].w << "\n";
    }
    //cout << "KECOM : " << kineticEnergiesVec[numTempGroups] << "\n";
//    for (int i = 0; i < context.getSystem().getNumParticles(); i++) {
//        normVel[i] = vel[i] - molVel[integrator.getParticleResId(i)];
//    }

//    clock_gettime(CLOCK_REALTIME, &t2); // Works on Linux
    // Add kinetic energy of ordinary particles.
    for (int i = 0; i < (int) normalParticleVec.size(); i++) {
        int index = normalParticleVec[i];
        Vec3 normVel = Vec3(normVelmVec[index].x,normVelmVec[index].y,normVelmVec[index].z);
        if (normVelmVec[index].w != 0) {
            kineticEnergiesVec[particleTempGroupVec[index]] += (normVel.dot(normVel))/normVelmVec[index].w;
        }
    }

    // Add kinetic energy of Drude particle pairs.
    //cout << "num normal particles : " << normalParticleVec.size() << "\n";
    //cout << "num pair particles : " << (int) pairParticleVec.size() << "\n";
    for (int i = 0; i < (int) pairParticleVec.size(); i++) {
        int p1 = pairParticleVec[i].x;
        int p2 = pairParticleVec[i].y;
        Vec3 normVel1 = Vec3(normVelmVec[p1].x,normVelmVec[p1].y,normVelmVec[p1].z);
        Vec3 normVel2 = Vec3(normVelmVec[p2].x,normVelmVec[p2].y,normVelmVec[p2].z);
        //double m1 = context.getSystem().getParticleMass(p1);
        //double m2 = context.getSystem().getParticleMass(p2);
        double m1 = 1.0/normVelmVec[p1].w;
        double m2 = 1.0/normVelmVec[p2].w;
        double totalMass = m1+m2;
        double reducedMass = m1*m2/totalMass;
        double mass1fract = m1/totalMass;
        double mass2fract = m2/totalMass;
        Vec3 cmVel = normVel1*mass1fract+normVel2*mass2fract;
        Vec3 relVel = normVel1-normVel2;
        kineticEnergiesVec[particleTempGroupVec[p1]] += (cmVel.dot(cmVel))*totalMass;
        kineticEnergiesVec[numTempGroups+1] += (relVel.dot(relVel))*reducedMass;
    }
    /*cout << "KEREAL : " << kineticEnergiesVec[0] << ", KECOM : " << kineticEnergiesVec[numTempGroups] << ", KEDRUDE : " << kineticEnergiesVec[numTempGroups+1] << "\n";
    for (int i = 0; i < vscaleFactorsVec.size(); i++) {
        cout << "vscale factor[" << i << "] : " << vscaleFactorsVec[i] << "\n";
        cout << "KE" << i << " : " << kineticEnergiesVec[i] << ", tempnkbt : " << tempGroupNkbT[i] << ", kediff : " << kineticEnergiesVec[i] - tempGroupNkbT[i] << "\n";
    }
    */

//    cout << "outer calc normal : " << normalKE << ", real : " << realKE << "\n";
//    cout << "kineticeEnergies : " << normalKE+realKE << ", " << drudeKE << "\n";
//    cout << "Before NHChain real T : " << (normalKE+realKE)/noseHooverDof.x/BOLTZ << ", drude T : " << drudeKE/noseHooverDof.y/BOLTZ << "\n";

//    clock_gettime(CLOCK_REALTIME, &t3); // Works on Linux
//    cout << "duration : " <<  (t3.tv_nsec - t2.tv_nsec ) << '\n';

    // Calculate scaling factor for velocities using multiple Nose-Hoover chain thermostat scheme
    vector<double> expfac(numTempGroups+2, 1.0);
    for (int itg = 0; itg < numTempGroups+1; itg++) {
        etaDotDot[itg][0] = (kineticEnergiesVec[itg] - tempGroupNkbT[itg]) / etaMass[itg][0];
        //cout << "itg : " << itg << ", etaDotDot : " << etaDotDot[itg][0] << ", etaDot : " << etaDot[itg][0] << "\n";
        for (int iter = 0; iter < numDrudeSteps; iter++) {
            for (int i = integrator.getNumNHChains()-1; i >= 0; i--) {
                expfac[itg] = exp(-dtc8 * etaDot[itg][i+1]);
                etaDot[itg][i] *= expfac[itg];
                etaDot[itg][i] += etaDotDot[itg][i] * dtc4;
                etaDot[itg][i] *= expfac[itg];
            }

            vscaleFactorsVec[itg] *= exp(-dtc2 * etaDot[itg][0]);
            kineticEnergiesVec[itg] *= exp(-dtc * etaDot[itg][0]);
            for (int i = 0; i < integrator.getNumNHChains(); i++) {
                eta[itg][i] += dtc2 * etaDot[itg][i];
            }

            etaDotDot[itg][0] = (kineticEnergiesVec[itg] - tempGroupNkbT[itg]) / etaMass[itg][0];

            etaDot[itg][0] *= expfac[itg];
            etaDot[itg][0] += etaDotDot[itg][0] * dtc4;
            etaDot[itg][0] *= expfac[itg];
            for (int i = 1; i < integrator.getNumNHChains(); i++) {
                expfac[itg] = exp(-dtc8 * etaDot[itg][i+1]);
                etaDot[itg][i] *= expfac[itg];
                etaDotDot[itg][i] = (etaMass[itg][i-1] * etaDot[itg][i-1] * etaDot[itg][i-1] - realkbT) / etaMass[itg][i];
                etaDot[itg][i] += etaDotDot[itg][i] * dtc4;
                etaDot[itg][i] *= expfac[itg];
            }
        }
        //cout << "itg : " << itg << ", etaDotDot : " << etaDotDot[itg][0] << ", etaDot : " << etaDot[itg][0] << ", dtc2 : " << dtc2 << ", scale : " << exp(-dtc2 * etaDot[itg][0]) << "\n";
    }

    int itg = numTempGroups+1;
//    for (int i = 0; i < numTempGroups+1; i++) {
//        cout << "KE" << i << " : " << kineticEnergiesVec[i] << "\n";
//    }
//    for (int i = 0; i < vscaleFactorsVec.size(); i++) {
//        cout << "vscale factor[" << i << "] : " << vscaleFactorsVec[i] << "\n";
//        cout << "KE" << i << " : " << kineticEnergiesVec[i] << ", tempnkbt : " << tempGroupNkbT[i] << ", kediff : " << kineticEnergiesVec[i] - tempGroupNkbT[i] << "\n";
//    }
    etaDotDot[itg][0] = (kineticEnergiesVec[itg] - tempGroupNkbT[itg]) / etaMass[itg][0];
    for (int iter = 0; iter < numDrudeSteps; iter++) {
        if (integrator.getUseDrudeNHChains()) {
            for (int i = integrator.getNumNHChains()-1; i > 0; i--) {
                expfac[itg] = exp(-dtc8 * etaDot[itg][i+1]);
                etaDot[itg][i] *= expfac[itg];
                etaDot[itg][i] += etaDotDot[itg][i] * dtc4;
                etaDot[itg][i] *= expfac[itg];
            }
        }
        expfac[itg] = exp(-dtc8 * etaDot[itg][1]);
        etaDot[itg][0] *= expfac[itg];
        etaDot[itg][0] += etaDotDot[itg][0] * dtc4;
        etaDot[itg][0] *= expfac[itg];

        vscaleFactorsVec[itg] *= exp(-dtc2 * etaDot[itg][0]);
        kineticEnergiesVec[itg] *= exp(-dtc * etaDot[itg][0]);

        eta[itg][0] += dtc2 * etaDot[itg][0];
        if (integrator.getUseDrudeNHChains()) {
            for (int i = 1; i < integrator.getNumNHChains(); i++) {
                eta[itg][i] += dtc2 * etaDot[itg][i];
            }
        }
        etaDotDot[itg][0] = (kineticEnergiesVec[itg] - tempGroupNkbT[itg]) / etaMass[itg][0];
        etaDot[itg][0] *= expfac[itg];
        etaDot[itg][0] += etaDotDot[itg][0] * dtc4;
        etaDot[itg][0] *= expfac[itg];
        if (integrator.getUseDrudeNHChains()) {
            for (int i = 1; i < integrator.getNumNHChains(); i++) {
                expfac[itg] = exp(-dtc8 * etaDot[itg][i+1]);
                etaDot[itg][i] *= expfac[itg];
                etaDotDot[itg][i] = (etaMass[itg][i-1] * etaDot[itg][i-1] * etaDot[itg][i-1] - drudekbT) / etaMass[itg][i];
                etaDot[itg][i] += etaDotDot[itg][i] * dtc4;
                etaDot[itg][i] *= expfac[itg];
            }
        }
    }
        //cout << "itg : " << itg << ", etaDotDot : " << etaDotDot[itg][0] << ", etaDot : " << etaDot[itg][0] << ", dtc2 : " << dtc2 << ", scale : " << exp(-dtc2 * etaDot[itg][0]) << "\n";


//    for (int i = 0; i < vscaleFactorsVec.size(); i++) {
//        cout << "vscale factor[" << i << "] : " << vscaleFactorsVec[i] << "\n";
//    }
//    cout << "After NHChain real T : " << kineticEnergiesVec[0]/noseHooverDof.x/BOLTZ << ", drude T : " << kineticEnergiesVec[1]/noseHooverDof.y/BOLTZ << "\n";
    return vscaleFactorsVec;
//std::make_tuple(vscale, vscaleDrude, vscaleCenter);
}

double CudaIntegrateDrudeNoseHooverStepKernel::computeKineticEnergy(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator) {
    return cu.getIntegrationUtilities().computeKineticEnergy(0.5*integrator.getStepSize());
}
