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
    if (kineticEnergies != NULL)
        delete kineticEnergies;
    if (normalKEBuffer != NULL)
        delete normalKEBuffer;
    if (realKEBuffer != NULL)
        delete realKEBuffer;
    if (drudeKEBuffer != NULL)
        delete drudeKEBuffer;
}

void CudaIntegrateDrudeNoseHooverStepKernel::initialize(const System& system, const DrudeNoseHooverIntegrator& integrator, const DrudeForce& force) {
    cu.getPlatformData().initializeContexts(system);

    noseHooverDof = make_int2(0, 0);
    noseHooverkbT = make_double2(BOLTZ * integrator.getTemperature(), BOLTZ * integrator.getDrudeTemperature());
    int   numDrudeSteps = integrator.getDrudeStepsPerRealStep();


    // Identify particle pairs and ordinary particles.
    
    set<int> particles;
    vector<int> normalParticleVec;
    vector<int2> pairParticleVec;
    for (int i = 0; i < system.getNumParticles(); i++) {
        particles.insert(i);
        double mass = system.getParticleMass(i);
        noseHooverDof.x += (mass == 0.0 ? 0 : 3);
    }
    for (int i = 0; i < force.getNumParticles(); i++) {
        int p, p1, p2, p3, p4;
        double charge, polarizability, aniso12, aniso34;
        force.getParticleParameters(i, p, p1, p2, p3, p4, charge, polarizability, aniso12, aniso34);
        particles.erase(p);
        particles.erase(p1);
        pairParticleVec.push_back(make_int2(p, p1));
        noseHooverDof.x -= 3;
        noseHooverDof.y += 3;
    }
    normalParticleVec.insert(normalParticleVec.begin(), particles.begin(), particles.end());
    normalParticles = CudaArray::create<int>(cu, max((int) normalParticleVec.size(), 1), "drudeNormalParticles");
    pairParticles = CudaArray::create<int2>(cu, max((int) pairParticleVec.size(), 1), "drudePairParticles");
    if (normalParticleVec.size() > 0)
        normalParticles->upload(normalParticleVec);
    if (pairParticleVec.size() > 0)
        pairParticles->upload(pairParticleVec);

    // reduce real d.o.f by number of constraints, and 3 if CMMotion remove is true
    noseHooverDof.x -= system.getNumConstraints();
    for (int i = 0; i < system.getNumForces(); i++) {
        cout << typeid(system.getForce(i)).name() << "\n";
        if (typeid(system.getForce(i)) == typeid(CMMotionRemover)) {
            cout << "CMMotion removal found, reduce dof by 3\n";
            noseHooverDof.x -= 3;
            break;
        }
    }

    // calculate etaMass
    noseHooverNkbT.x = noseHooverDof.x * noseHooverkbT.x;
    noseHooverNkbT.y = noseHooverDof.y * noseHooverkbT.y;
    double realEtaMass = noseHooverNkbT.x * pow(integrator.getCouplingTime(), 2);        // COM
    double drudeEtaMass = noseHooverNkbT.y * pow(integrator.getDrudeCouplingTime(), 2);  // internal
    etaMass.push_back(make_double2(realEtaMass, drudeEtaMass));

    cout << "Initialization finished\n";
    cout << "real T : " << integrator.getTemperature() << ", drude T : " << integrator.getDrudeTemperature() << "\n";
    cout << "real NkbT : " << noseHooverNkbT.x << ", drude NkbT : " << noseHooverNkbT.y << ", etaMass : " << etaMass[0].x;
    cout << ", drudeQ0 : " << etaMass[0].y << "\n";
    cout << "real Dof : " << noseHooverDof.x << ", drude Dof : " << noseHooverDof.y << "\n";
    cout << "Num NH Chain : " << integrator.getNumNHChains() << "\n";
    cout << "Use NH Chain for Drude dof : " << integrator.getUseDrudeNHChains() << "\n";
    cout << "real couplingTime : " << integrator.getCouplingTime() << "\n";
    cout << "drude couplingTime : " << integrator.getDrudeCouplingTime() << "\n";
    cout << "pair Particles[0].x : " << pairParticleVec[0].x << "\n";
    cout << "pair Particles[0].y : " << pairParticleVec[0].y << "\n";

    // initialize eta values to zero
    eta.push_back(make_double2(0.0,0.0));
    etaDot.push_back(make_double2(0.0,0.0));
    etaDotDot.push_back(make_double2(0.0,0.0));
    realEtaMass = noseHooverkbT.x * pow(integrator.getCouplingTime(), 2);        // COM
    drudeEtaMass = noseHooverkbT.y * pow(integrator.getDrudeCouplingTime(), 2);  // internal
    for (int ich=1; ich < integrator.getNumNHChains(); ich++) {
        etaMass.push_back(make_double2(realEtaMass, drudeEtaMass));
        eta.push_back(make_double2(0.0,0.0));
        etaDot.push_back(make_double2(0.0,0.0));
        etaDotDot.push_back(make_double2(0.0,0.0));
        etaDotDot[ich].x = (etaMass[ich-1].x * etaDot[ich-1].x * etaDot[ich-1].x - noseHooverkbT.x) / etaMass[ich].x;
        if (integrator.getUseDrudeNHChains()) {
            etaDotDot[ich].y = (etaMass[ich-1].y * etaDot[ich-1].y * etaDot[ich-1].y - noseHooverkbT.y) / etaMass[ich].y;
        }
    }
    // extra dummy chain which will always have etaDot = 0
    etaDot.push_back(make_double2(0.0,0.0));


    // Create kernels.
    int elementSize = (cu.getUseDoublePrecision() || cu.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
    cout << "ElementSize : " << elementSize << ", double : " << sizeof(double) << ", float : " << sizeof(float) << "\n";
    kineticEnergies = new CudaArray(cu, 2, elementSize, "kineticEnergies");
    normalKEBuffer = new CudaArray(cu, max((int) normalParticleVec.size(),1), elementSize, "normalKEBuffer");
    realKEBuffer = new CudaArray(cu, max((int) pairParticleVec.size(),1), elementSize, "realKEBuffer");
    drudeKEBuffer = new CudaArray(cu, max((int) pairParticleVec.size(),1), elementSize, "drudeKEBuffer");
    
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_NORMAL_PARTICLES"] = cu.intToString(normalParticleVec.size());
    defines["NUM_PAIRS"] = cu.intToString(pairParticleVec.size());
    defines["WORK_GROUP_SIZE"] = cu.intToString(CudaContext::ThreadBlockSize);
    cout << "NUM_ATOMS : " << cu.getNumAtoms() << ", PADDED_NUM_ATOMS : " << cu.getPaddedNumAtoms() << ", NUM_NORMAL_PARTICLES : " << normalParticleVec.size() << ", NUM_PAIRS : " << pairParticleVec.size() << ", WORK_GROUP_SIZE : " << CudaContext::ThreadBlockSize << "\n";
    //defines["NUM_DRUDE_STEPS"] = cu.intToString(numDrudeSteps);
    map<string, string> replacements;
    CUmodule module = cu.createModule(CudaDrudeNoseKernelSources::vectorOps+CudaDrudeNoseKernelSources::drudeNoseHoover, defines, "");
    kernelKE = cu.getKernel(module, "computeDrudeNoseHooverKineticEnergies");
    kernelKESum = cu.getKernel(module, "sumDrudeKineticEnergies");
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
    
    // Compute integrator coefficients.
    
    double stepSize = integrator.getStepSize();
    //cout << "stepSize : " << stepSize << "\n";

    double vscale = 1.0;
    double fscale = 0.5*stepSize/(double) 0x100000000;
    double vscaleDrude = 1.0;
    double fscaleDrude = fscale;
    double vscaleNone = 1.0;
    double fscaleZero = 0.0;
    double vscaleDrudeNone = 1.0;
    double fscaleDrudeZero = 0.0;
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
    
    float vscaleFloat = (float) vscale;
    float fscaleFloat = (float) fscale;
    float vscaleDrudeFloat = (float) vscaleDrude;
    float fscaleDrudeFloat = (float) fscaleDrude;
    float vscaleNoneFloat = (float) vscaleNone;
    float fscaleZeroFloat = (float) fscaleZero;
    float vscaleDrudeNoneFloat = (float) vscaleDrudeNone;
    float fscaleDrudeZeroFloat = (float) fscaleDrudeZero;
    float maxDrudeDistanceFloat =(float) maxDrudeDistance;
    float hardwallscaleDrudeFloat = (float) hardwallscaleDrude;
    void *vscalePtr, *fscalePtr, *vscaleDrudePtr, *fscaleDrudePtr, *maxDrudeDistancePtr, *hardwallscaleDrudePtr;
    void *vscaleNonePtr, *fscaleZeroPtr, *vscaleDrudeNonePtr, *fscaleDrudeZeroPtr;
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        vscalePtr = &vscale;
        fscalePtr = &fscale;
        vscaleDrudePtr = &vscaleDrude;
        fscaleDrudePtr = &fscaleDrude;
        vscaleNonePtr = &vscaleNone;
        fscaleZeroPtr = &fscaleZero;
        vscaleDrudeNonePtr = &vscaleDrudeNone;
        fscaleDrudeZeroPtr = &fscaleDrudeZero;
        maxDrudeDistancePtr = &maxDrudeDistance;
        hardwallscaleDrudePtr = &hardwallscaleDrude;
    }
    else {
        vscalePtr = &vscaleFloat;
        fscalePtr = &fscaleFloat;
        vscaleDrudePtr = &vscaleDrudeFloat;
        fscaleDrudePtr = &fscaleDrudeFloat;
        vscaleNonePtr = &vscaleNoneFloat;
        fscaleZeroPtr = &fscaleZeroFloat;
        vscaleDrudeNonePtr = &vscaleDrudeNoneFloat;
        fscaleDrudeZeroPtr = &fscaleDrudeZeroFloat;
        maxDrudeDistancePtr = &maxDrudeDistanceFloat;
        hardwallscaleDrudePtr = &hardwallscaleDrudeFloat;
    }

    // First half of velocity + thermostat integration.
    double2 vscaleVec = propagateNHChain(context, integrator);
    vscale = vscaleVec.x;
    vscaleDrude = vscaleVec.y;
    vscaleFloat = (float) vscale;
    vscaleDrudeFloat = (float) vscaleDrude;

    //cout << "vscale : " << vscale << ", vscaleDrude : " << vscaleDrude << "\n";


    bool updatePosDelta = false;
    void *updatePosDeltaPtr = &updatePosDelta;
    void* argsChain[] = {&cu.getVelm().getDevicePointer(),
            &normalParticles->getDevicePointer(), &pairParticles->getDevicePointer(), &integration.getStepSize().getDevicePointer(),
            vscalePtr, vscaleDrudePtr};
    cu.executeKernel(kernelChain, argsChain, numAtoms);

    // Call the first half of velocity integration kernel. (both thermostat and actual velocity update)
    updatePosDelta = true;
    void* argsVel[] = {&cu.getVelm().getDevicePointer(), &cu.getForce().getDevicePointer(), &integration.getPosDelta().getDevicePointer(),
            &normalParticles->getDevicePointer(), &pairParticles->getDevicePointer(), &integration.getStepSize().getDevicePointer(),
            fscalePtr, fscaleDrudePtr, updatePosDeltaPtr};
    //        vscaleNonePtr, fscalePtr, vscaleDrudeNonePtr, fscaleDrudePtr, updatePosDeltaPtr};
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
    //        vscaleNonePtr, fscalePtr, vscaleDrudeNonePtr, fscaleDrudePtr, updatePosDeltaPtr};
    cu.executeKernel(kernelVel, argsVel2, numAtoms);

    // Apply velocity constraints
    integration.applyVelocityConstraints(integrator.getConstraintTolerance());

    // Second half of thermostat integration.
    vscaleVec = propagateNHChain(context, integrator);
    vscale = vscaleVec.x;
    vscaleDrude = vscaleVec.y;
    vscaleFloat = (float) vscale;
    vscaleDrudeFloat = (float) vscaleDrude;

    // Call the second half of velocity integration kernel. (Nose-Hoover chain thermostat update only)
    cu.executeKernel(kernelChain, argsChain, numAtoms);

    // Update the time and step count.
    cu.setTime(cu.getTime()+stepSize);
    cu.setStepCount(cu.getStepCount()+1);
    cu.reorderAtoms();
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables
------------------------------------------------------------------------- */
double2 CudaIntegrateDrudeNoseHooverStepKernel::propagateNHChain(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator) {
    int numAtoms = cu.getNumAtoms();

    double stepSize = integrator.getStepSize();
    int    numDrudeSteps = integrator.getDrudeStepsPerRealStep();
    double dtc = stepSize/numDrudeSteps;
    double dtc2 = dtc/2.0;
    double dtc4 = dtc/4.0;
    double dtc8 = dtc/8.0;
    double vscale = 1.0;
    double vscaleDrude = 1.0;

    vector<double> kineticEnergiesVec(2);
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

    double normalKE = 0.0;
    double realKE = 0.0;
    double drudeKE = 0.0;
    vector<Vec3> vel;
    State state;
    state = context.getOwner().getState(State::Velocities);
    vel = state.getVelocities();
    vector<int> normalParticleVec;
    vector<int2> pairParticleVec;

    normalParticles->download(normalParticleVec);
    pairParticles->download(pairParticleVec);

//    clock_gettime(CLOCK_REALTIME, &t2); // Works on Linux
    // Add kinetic energy of ordinary particles.
    for (int i = 0; i < (int) normalParticleVec.size(); i++) {
        int index = normalParticleVec[i];
        double m1 =  context.getSystem().getParticleMass(index);
        if (m1 != 0) {
            normalKE += (vel[index].dot(vel[index]))*m1;
        }
    }

    // Add kinetic energy of Drude particle pairs.
    for (int i = 0; i < (int) pairParticleVec.size(); i++) {
        int p1 = pairParticleVec[i].x;
        int p2 = pairParticleVec[i].y;
        double m1 = context.getSystem().getParticleMass(p1);
        double m2 = context.getSystem().getParticleMass(p2);
        double totalMass = m1+m2;
        double reducedMass = m1*m2/totalMass;
        double mass1fract = m1/totalMass;
        double mass2fract = m2/totalMass;
        Vec3 cmVel = vel[p1]*mass1fract+vel[p2]*mass2fract;
        Vec3 relVel = vel[p2]-vel[p1];
        realKE += (cmVel.dot(cmVel))*totalMass;
        drudeKE += (relVel.dot(relVel))*reducedMass;
    }

//    cout << "outer calc normal : " << normalKE << ", real : " << realKE << "\n";
//    cout << "kineticeEnergies : " << normalKE+realKE << ", " << drudeKE << "\n";
//    cout << "Before NHChain real T : " << (normalKE+realKE)/noseHooverDof.x/BOLTZ << ", drude T : " << drudeKE/noseHooverDof.y/BOLTZ << "\n";
    kineticEnergiesVec[0] = normalKE+realKE;
    kineticEnergiesVec[1] = drudeKE;

//    clock_gettime(CLOCK_REALTIME, &t3); // Works on Linux
//    cout << "duration : " <<  (t3.tv_nsec - t2.tv_nsec ) << '\n';

    // Calculate scaling factor for velocities using multiple Nose-Hoover chain thermostat scheme
    double2 expfac = make_double2(1.0, 1.0);
    etaDotDot[0].x = (kineticEnergiesVec[0] - noseHooverNkbT.x) / etaMass[0].x;
    etaDotDot[0].y = (kineticEnergiesVec[1] - noseHooverNkbT.y) / etaMass[0].y;
    for (int iter = 0; iter < numDrudeSteps; iter++) {
        for (int i = integrator.getNumNHChains()-1; i >= 0; i--) {
            expfac.x = exp(-dtc8 * etaDot[i+1].x);
            etaDot[i].x *= expfac.x;
            etaDot[i].x += etaDotDot[i].x * dtc4;
            etaDot[i].x *= expfac.x;
            if (integrator.getUseDrudeNHChains() or i==0) {
                expfac.y = exp(-dtc8 * etaDot[i+1].y);
                etaDot[i].y *= expfac.y;
                etaDot[i].y += etaDotDot[i].y * dtc4;
                etaDot[i].y *= expfac.y;
            }
        }

        vscale *= exp(-dtc2 * etaDot[0].x);
        vscaleDrude *= exp(-dtc2 * etaDot[0].y);
        kineticEnergiesVec[0] *= exp(-dtc * etaDot[0].x);
        kineticEnergiesVec[1] *= exp(-dtc * etaDot[0].y);
        for (int i = 0; i < integrator.getNumNHChains(); i++) {
            eta[i].x += dtc2 * etaDot[i].x;
            if (integrator.getUseDrudeNHChains() or i==0) {
                eta[i].y += dtc2 * etaDot[i].y;
            }
        }

        etaDotDot[0].x = (kineticEnergiesVec[0] - noseHooverNkbT.x) / etaMass[0].x;
        etaDotDot[0].y = (kineticEnergiesVec[1] - noseHooverNkbT.y) / etaMass[0].y;

        for (int i = 0; i < integrator.getNumNHChains(); i++) {
            expfac.x = exp(-dtc8 * etaDot[i+1].x);
            etaDot[i].x *= expfac.x;
            if (i > 0) {
                etaDotDot[i].x = (etaMass[i-1].x * etaDot[i-1].x * etaDot[i-1].x - noseHooverkbT.x) / etaMass[i].x;
            }
            etaDot[i].x += etaDotDot[i].x * dtc4;
            etaDot[i].x *= expfac.x;
            if (integrator.getUseDrudeNHChains() or i==0) {
                expfac.y = exp(-dtc8 * etaDot[i+1].y);
                etaDot[i].y *= expfac.y;
                if (i > 0) {
                    etaDotDot[i].y = (etaMass[i-1].y * etaDot[i-1].y * etaDot[i-1].y - noseHooverkbT.y) / etaMass[i].y;
                }
                etaDot[i].y += etaDotDot[i].y * dtc4;
                etaDot[i].y *= expfac.y;
            }
        }
    }

//    cout << "After NHChain real T : " << kineticEnergiesVec[0]/noseHooverDof.x/BOLTZ << ", drude T : " << kineticEnergiesVec[1]/noseHooverDof.y/BOLTZ << "\n";
    return make_double2(vscale, vscaleDrude);
}

double CudaIntegrateDrudeNoseHooverStepKernel::computeKineticEnergy(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator) {
    return cu.getIntegrationUtilities().computeKineticEnergy(0.5*integrator.getStepSize());
}
