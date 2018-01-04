/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2011-2014 Stanford University and the Authors.      *
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

#include "ReferenceDrudeNoseKernels.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/System.h"
#include "openmm/OpenMMException.h"
#include "openmm/CMMotionRemover.h"
#include "openmm/internal/ContextImpl.h"
#include "SimTKOpenMMUtilities.h"
#include "ReferenceConstraints.h"
#include "ReferenceVirtualSites.h"
#include <algorithm>
#include <set>
#include <typeinfo>
#include <iostream>


using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->velocities);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

static ReferenceConstraints& extractConstraints(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *(ReferenceConstraints*) data->constraints;
}

static double computeShiftedKineticEnergy(ContextImpl& context, vector<double>& inverseMasses, double timeShift) {
    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    vector<RealVec>& posData = extractPositions(context);
    vector<RealVec>& velData = extractVelocities(context);
    vector<RealVec>& forceData = extractForces(context);
    
    // Compute the shifted velocities.
    
    vector<RealVec> shiftedVel(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        if (inverseMasses[i] > 0)
            shiftedVel[i] = velData[i]+forceData[i]*(timeShift*inverseMasses[i]);
        else
            shiftedVel[i] = velData[i];
    }
    
    // Apply constraints to them.
    
    extractConstraints(context).applyToVelocities(posData, shiftedVel, inverseMasses, 1e-4);
    
    // Compute the kinetic energy.
    
    double energy = 0.0;
    for (int i = 0; i < numParticles; ++i)
        if (inverseMasses[i] > 0)
            energy += (shiftedVel[i].dot(shiftedVel[i]))/inverseMasses[i];
    return 0.5*energy;
}


ReferenceIntegrateDrudeNoseHooverStepKernel::~ReferenceIntegrateDrudeNoseHooverStepKernel() {
}

void ReferenceIntegrateDrudeNoseHooverStepKernel::initialize(const System& system, const DrudeNoseHooverIntegrator& integrator, const DrudeForce& force) {
    realDof = 0;
    drudeDof = 0;
    centerDof = 0;
    realkbT = BOLTZ * integrator.getTemperature();
    drudekbT = BOLTZ * integrator.getDrudeTemperature();
    centerkbT = BOLTZ * integrator.getTemperature();

    
    // Identify particle pairs and ordinary particles.
    
    set<int> particles;
    for (int i = 0; i < system.getNumParticles(); i++) {
        particles.insert(i);
        double mass = system.getParticleMass(i);
        particleMass.push_back(mass);
        particleInvMass.push_back(mass == 0.0 ? 0.0 : 1.0/mass);
        realDof += (mass == 0.0 ? 0 : 3);
    }
    // center particles use independent Dof and temperature group
    for (int i=0; i < integrator.getNumCenterParticles(); i++) {
        int p;
        integrator.getCenterParticle(i, p);
        centerParticles.push_back(p);
    }
    for (int i = 0; i < force.getNumParticles(); i++) {
        int p, p1, p2, p3, p4;
        double charge, polarizability, aniso12, aniso34;
        force.getParticleParameters(i, p, p1, p2, p3, p4, charge, polarizability, aniso12, aniso34);
        particles.erase(p);
        particles.erase(p1);
        pairParticles.push_back(make_pair(p, p1));

        // check if the particle is center particle
        if ( ( std::find(centerParticles.begin(), centerParticles.end(), p) != centerParticles.end() ) or ( std::find(centerParticles.begin(), centerParticles.end(), p1) != centerParticles.end() ) ) {
            pairIsCenterParticle.push_back(true);
            realDof -= 3;
            centerDof += 3;
        }
        else
            pairIsCenterParticle.push_back(false);

        double m1 = system.getParticleMass(p);
        double m2 = system.getParticleMass(p1);
        pairInvTotalMass.push_back(1.0/(m1+m2));
        pairInvReducedMass.push_back((m1+m2)/(m1*m2));
        realDof -= 3;
        drudeDof += 3;
    }

    normalParticles.insert(normalParticles.begin(), particles.begin(), particles.end());

    if (integrator.getUseDrudeNHChains()) {
        if (centerDof > 0)
            numTempGroup = 3;
        else
            numTempGroup = 2;
    }
    else {
        if (centerDof > 0)
            numTempGroup = 2;
        else
            numTempGroup = 1;
    }
            
    iNumNHChains = integrator.getNumNHChains();
    if (integrator.getUseDrudeNHChains()) {
        idxMaxNHChains = integrator.getNumNHChains()*numTempGroup-1;
        iNumNHChains = iNumNHChains * numTempGroup;
    }
    else {
        idxMaxNHChains = integrator.getNumNHChains()*numTempGroup;
        iNumNHChains = iNumNHChains * numTempGroup + 1;
    }

    // reduce real d.o.f by number of constraints, and 3 if CMMotion remove is true
    realDof -= system.getNumConstraints();
    for (int i = 0; i < system.getNumForces(); i++) {
        cout << typeid(system.getForce(i)).name() << "\n";
        if (typeid(system.getForce(i)) == typeid(CMMotionRemover)) {
            cout << "CMMotion removal found, reduce dof by 3\n";
            realDof -= 3;
            break;
        }
    }

    // calculate etaMass and etaMass
    realNkbT = realDof * realkbT;
    drudeNkbT = drudeDof * drudekbT;
    centerNkbT = centerDof * centerkbT;
    etaMass.push_back(realNkbT * pow(integrator.getCouplingTime(), 2));             // COM
    etaMass.push_back(drudeNkbT * pow(integrator.getDrudeCouplingTime(), 2));       // internal

    cout << "Initialization finished\n";
    cout << "real T : " << integrator.getTemperature() << ", drude T : " << integrator.getDrudeTemperature() << "\n";
    cout << "realNkbT : " << realNkbT << ", drudeNkbT : " << drudeNkbT << ", centerNkbT : " << centerNkbT << "\n";
    cout << "etaMass : " << etaMass[0] << ", drudeQ0 : " << etaMass[1] << "\n";
    cout << "realDof : " << realDof << ", centerDof : " << centerDof << ", drudeDof : " << drudeDof << "\n";
    cout << "Num NH Chain : " << integrator.getNumNHChains() << "\n";
    cout << "real couplingTime : " << integrator.getCouplingTime() << "\n";
    cout << "drude couplingTime : " << integrator.getDrudeCouplingTime() << "\n";
    cout << "pair Particles[0].first : " << pairParticles[0].first << "\n";
    cout << "pair Particles[0].second : " << pairParticles[0].second << "\n";

    // Loop over number of temperature groups 
    // initialize eta values to zero
    eta.push_back(0.0);         // COM
    eta.push_back(0.0);         // internal
    etaDot.push_back(0.0);      // COM
    etaDot.push_back(0.0);      // internal
    etaDotDot.push_back(0.0);   // COM
    etaDotDot.push_back(0.0);   // internal
    // Improper temperature group
    if (centerDof > 0) {
        etaMass.push_back(centerNkbT * pow(integrator.getCouplingTime(), 2));           // center
        cout << "There's center particles !!\n  centerMass : " << etaMass[2] << "\n";
        eta.push_back(0.0);         // center
        etaDot.push_back(0.0);      // center
        etaDotDot.push_back(0.0);   // center
        numTempGroup = 3;
    }
    if (integrator.getUseDrudeNHChains()) {
        for (int ich=1; ich < integrator.getNumNHChains(); ich++) {
            etaMass.push_back(realkbT * pow(integrator.getCouplingTime(), 2));        // COM
            etaMass.push_back(drudekbT * pow(integrator.getDrudeCouplingTime(), 2));   // internal
            eta.push_back(0.0);             // COM
            eta.push_back(0.0);             // internal
            etaDot.push_back(0.0);          // COM
            etaDot.push_back(0.0);          // internal
            etaDotDot.push_back(0.0);       // COM
            etaDotDot.push_back(0.0);       // internal
            etaDotDot[ich*numTempGroup] = (etaMass[(ich-1)*numTempGroup] * etaDot[(ich-1)*numTempGroup] * etaDot[(ich-1)*numTempGroup] - realkbT) / etaMass[ich*numTempGroup];
            etaDotDot[ich*numTempGroup+1] = (etaMass[(ich-1)*numTempGroup+1] * etaDot[(ich-1)*numTempGroup+1] * etaDot[(ich-1)*numTempGroup+1] - drudekbT) / etaMass[ich*numTempGroup+1];
            // Improper temperature group
            if (centerDof > 0) {
                etaMass.push_back(centerkbT * pow(integrator.getCouplingTime(), 2));           // center
                eta.push_back(0.0);         // center
                etaDot.push_back(0.0);      // center
                etaDotDot.push_back(0.0);   // center
                etaDotDot[ich*numTempGroup+2] = (etaMass[(ich-1)*numTempGroup+2] * etaDot[(ich-1)*numTempGroup+2] * etaDot[(ich-1)*numTempGroup+2] - centerkbT) / etaMass[ich*numTempGroup+2];
            }
        }
    }
    else {
        for (int ich=1; ich < integrator.getNumNHChains(); ich++) {
            etaMass.push_back(realkbT * pow(integrator.getCouplingTime(), 2));        // COM
            eta.push_back(0.0);             // COM
            etaDot.push_back(0.0);          // COM
            etaDotDot.push_back(0.0);       // COM
            etaDotDot[ich*numTempGroup+1] = (etaMass[(ich-1)*numTempGroup+1] * etaDot[(ich-1)*numTempGroup+1] * etaDot[(ich-1)*numTempGroup+1] - realkbT) / etaMass[ich*numTempGroup+1];
            // Improper temperature group
            if (centerDof > 0) {
                etaMass.push_back(centerkbT * pow(integrator.getCouplingTime(), 2));           // center
                eta.push_back(0.0);         // center
                etaDot.push_back(0.0);      // center
                etaDotDot.push_back(0.0);   // center
                etaDotDot[ich*numTempGroup+2] = (etaMass[(ich-1)*numTempGroup+2] * etaDot[(ich-1)*numTempGroup+2] * etaDot[(ich-1)*numTempGroup+2] - realkbT) / etaMass[ich*numTempGroup+2];
            }
        }
    }
    // extra dummy chain which will always have etaDot = 0
    etaDot.push_back(0.0);          // COM
    etaDot.push_back(0.0);          // internal
    // Improper temperature group
    if (centerDof > 0) {
        etaDot.push_back(0.0);          // center
        cout << "realMass1 : " << etaMass[3] << ", drudeQ1 : " << etaMass[4] << ", centerMass1 : " << etaMass[5] << "\n";
    }
    else {
        cout << "realMass1 : " << etaMass[2] << ", drudeQ1 : " << etaMass[3] << "\n";
    }
}

void ReferenceIntegrateDrudeNoseHooverStepKernel::execute(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator) {
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& vel = extractVelocities(context);
    vector<RealVec>& force = extractForces(context);

    //cout << "\n Debug: Befor anyting happens; first force element : \n";
    //cout << force[pairParticles[0].first] << "\n";
    //cout << "\n Debug: Befor anyting happens; first vel element : \n";
    //cout << vel[pairParticles[0].first];
    // First Nose-Hoover chain thermostat scheme
    propagateNHChain(context, integrator);

   // cout << "\n Debug: After first Nose Hoover Chain propagation; first force element : \n";
   // force = extractForces(context);
   // cout << force[pairParticles[0].first];
   // cout << "\n Debug: After first Nose Hoover Chain propagation; first vel element : \n";
   // cout << vel[pairParticles[0].first];
    // first propagation of half velocity
    propagateHalfVelocity(context, integrator);

    //cout << "\n Debug: After first propagation of VV2; first force element : \n";
    //force = extractForces(context);
    //cout << "compare to first velocity : " << vel[0] << ", second velocity : " << vel[1] << "\n";
    //State state = context.getOwner().getState(State::Velocities);
    //vector<Vec3> veltemp = state.getVelocities();
    //veltemp = state.getVelocities();
    //cout << "first velocity : " << veltemp[0] << ", second velocity : " << veltemp[1] << "\n";
    //cout << force[pairParticles[0].first] << "\n";
   // cout << "\n Debug: After first propagation of VV2; first vel element : \n";
   // cout << vel[pairParticles[0].first];
    // Update the particle positions.
    
    int numParticles = particleInvMass.size();
    vector<RealVec> xPrime(numParticles);
    RealOpenMM dt = integrator.getStepSize();
    for (int i = 0; i < numParticles; i++)
        if (particleInvMass[i] != 0.0)
            xPrime[i] = pos[i]+vel[i]*dt;
    
    //cout << "\n Debug: After position propagation of VV2; first force element : \n";
    //force = extractForces(context);
    //cout << "first velocity : " << vel[0] << ", second velocity : " << vel[1] << "\n";
   // cout << force[pairParticles[0].first];
   // cout << "\n Debug: After position propagation of VV2; first vel element : \n";
   // cout << vel[pairParticles[0].first];
    // Apply constraints.
    
    extractConstraints(context).apply(pos, xPrime, particleInvMass, integrator.getConstraintTolerance());
    
    //cout << "\n Debug: After constraints.apply; first force element : \n";
    //force = extractForces(context);
    //cout << "first velocity : " << vel[0] << ", second velocity : " << vel[1] << "\n";
   // cout << force[pairParticles[0].first];
   // cout << "\n Debug: After constraints.apply; first vel element : \n";
   // cout << vel[pairParticles[0].first];
    // Record the constrained positions and velocities.
    
    RealOpenMM dtInv = 1.0/dt;
    for (int i = 0; i < numParticles; i++) {
        if (particleInvMass[i] != 0.0) {
            vel[i] = (xPrime[i]-pos[i])*dtInv;
            pos[i] = xPrime[i];
        }
    }
    //cout << "\n Debug: After recording constrained pos & vel; first force element : \n";
    //force = extractForces(context);
    //cout << "first velocity : " << vel[0] << ", second velocity : " << vel[1] << "\n";
   // cout << force[pairParticles[0].first];
   // cout << "\n Debug: After recording constrained pos & vel; first vel element : \n";
   // cout << vel[pairParticles[0].first];
    //cout << "compare to first velocity : " << vel[0] << ", second velocity : " << vel[1] << "\n";
    //state = context.getOwner().getState(State::Velocities);
    //veltemp = state.getVelocities();
    //cout << "first velocity : " << veltemp[0] << ", second velocity : " << veltemp[1] << "\n";

    // Apply hard wall constraints.

    const RealOpenMM maxDrudeDistance = integrator.getMaxDrudeDistance();
    if (maxDrudeDistance > 0) {
        const RealOpenMM hardwallscaleDrude = sqrt(drudekbT);
        for (int i = 0; i < (int) pairParticles.size(); i++) {
            int p1 = pairParticles[i].first;
            int p2 = pairParticles[i].second;
            RealVec delta = pos[p1]-pos[p2];
            RealOpenMM r = sqrt(delta.dot(delta));
            RealOpenMM rInv = 1/r;
            if (rInv*maxDrudeDistance < 1.0) {
                // The constraint has been violated, so make the inter-particle distance "bounce"
                // off the hard wall.
                
                if (rInv*maxDrudeDistance < 0.5)
                    throw OpenMMException("Drude particle moved too far beyond hard wall constraint");
                RealVec bondDir = delta*rInv;
                RealVec vel1 = vel[p1];
                RealVec vel2 = vel[p2];
                RealOpenMM mass1 = particleMass[p1];
                RealOpenMM mass2 = particleMass[p2];
                RealOpenMM deltaR = r-maxDrudeDistance;
                RealOpenMM deltaT = dt;
                RealOpenMM dotvr1 = vel1.dot(bondDir);
                RealVec vb1 = bondDir*dotvr1;
                RealVec vp1 = vel1-vb1;
                if (mass2 == 0) {
                    // The parent particle is massless, so move only the Drude particle.

                    if (dotvr1 != 0.0)
                        deltaT = deltaR/abs(dotvr1);
                    if (deltaT > dt)
                        deltaT = dt;
                    dotvr1 = -dotvr1*hardwallscaleDrude/(abs(dotvr1)*sqrt(mass1));
                    RealOpenMM dr = -deltaR + deltaT*dotvr1;
                    pos[p1] += bondDir*dr;
                    vel[p1] = vp1 + bondDir*dotvr1;
                }
                else {
                    // Move both particles.

                    RealOpenMM invTotalMass = pairInvTotalMass[i];
                    RealOpenMM dotvr2 = vel2.dot(bondDir);
                    RealVec vb2 = bondDir*dotvr2;
                    RealVec vp2 = vel2-vb2;
                    RealOpenMM vbCMass = (mass1*dotvr1 + mass2*dotvr2)*invTotalMass;
                    dotvr1 -= vbCMass;
                    dotvr2 -= vbCMass;
                    if (dotvr1 != dotvr2)
                        deltaT = deltaR/abs(dotvr1-dotvr2);
                    if (deltaT > dt)
                        deltaT = dt;
                    RealOpenMM vBond = hardwallscaleDrude/sqrt(mass1);
                    dotvr1 = -dotvr1*vBond*mass2*invTotalMass/abs(dotvr1);
                    dotvr2 = -dotvr2*vBond*mass1*invTotalMass/abs(dotvr2);
                    RealOpenMM dr1 = -deltaR*mass2*invTotalMass + deltaT*dotvr1;
                    RealOpenMM dr2 = deltaR*mass1*invTotalMass + deltaT*dotvr2;
                    dotvr1 += vbCMass;
                    dotvr2 += vbCMass;
                    pos[p1] += bondDir*dr1;
                    pos[p2] += bondDir*dr2;
                    vel[p1] = vp1 + bondDir*dotvr1;
                    vel[p2] = vp2 + bondDir*dotvr2;
                }
            }
        }
    }
    //cout << "\n Debug: After applying hard-wall constraints; first force element : \n";
    //cout << "compare to first velocity : " << vel[0] << ", second velocity : " << vel[1] << "\n";
    //state = context.getOwner().getState(State::Velocities);
    //veltemp = state.getVelocities();
    //cout << "first velocity : " << veltemp[0] << ", second velocity : " << veltemp[1] << "\n";
   // force = extractForces(context);
   // cout << force[pairParticles[0].first];
   // cout << "\n Debug: After applying hard-wall constraints; first vel element : \n";
   // cout << vel[pairParticles[0].first];
    ReferenceVirtualSites::computePositions(context.getSystem(), pos);
    //cout << "\n Debug: After virtualsites::computePositions; first force element : \n";
    ////force = extractForces(context);
    //cout << force[pairParticles[0].first];
    //cout << "compare to first velocity : " << vel[0] << ", second velocity : " << vel[1] << "\n";
    //state = context.getOwner().getState(State::Velocities);
    //veltemp = state.getVelocities();
    //cout << "first velocity : " << veltemp[0] << ", second velocity : " << veltemp[1] << "\n";
   // cout << "\n Debug: After virtualsites::computePositions; first vel element : \n";
   // cout << vel[pairParticles[0].first];

    context.calcForcesAndEnergy(true, false);
    //cout << "\n Debug: After calcForcesAndEnergy; first force element : \n";
    //state = context.getOwner().getState(State::Velocities);
    //veltemp = state.getVelocities();
    //cout << "first velocity : " << veltemp[0] << ", second velocity : " << veltemp[1] << "\n";
    //force = extractForces(context);
    //cout << force[pairParticles[0].first];
    //cout << "\n Debug: After calcForcesAndEnergy; first vel element : \n";
    //cout << vel[pairParticles[0].first];
    // second propagation of half velocity
    propagateHalfVelocity(context, integrator);
    //cout << "\n Debug: After second velocity propagation of VV2; first force element : \n";
    //cout << "compare to first velocity : " << vel[0] << ", second velocity : " << vel[1] << "\n";
    //state = context.getOwner().getState(State::Velocities);
    //veltemp = state.getVelocities();
    //cout << "first velocity : " << veltemp[0] << ", second velocity : " << veltemp[1] << "\n";
   // force = extractForces(context);
   // cout << force[pairParticles[0].first];
   // cout << "\n Debug: After second velocity propagation of VV2; first vel element : \n";
   // cout << vel[pairParticles[0].first];

    // Second Nose-Hoover chain thermostat scheme
    propagateNHChain(context, integrator);
    //cout << "\n Debug: After second Nose Hoover Chain propagation of VV2; first force element : \n";
    //force = extractForces(context);
    //cout << force[pairParticles[0].first];
    //cout << "\n Debug: After second Nose Hoover Chain propagation of VV2; first vel element : \n";
    //cout << vel[pairParticles[0].first];

    data.time += integrator.getStepSize();
    data.stepCount++;
}

/* ----------------------------------------------------------------------
   compute kinetic energies for each degrees of freedom
------------------------------------------------------------------------- */
//void ReferenceIntegrateDrudeNoseHooverStepKernel::computeNHKineticEnergy(ContextImpl& context) {
//}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables
------------------------------------------------------------------------- */
void ReferenceIntegrateDrudeNoseHooverStepKernel::propagateNHChain(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator)
{
    vector<RealVec>& vel = extractVelocities(context);

    int   numDrudeSteps = integrator.getDrudeStepsPerRealStep();
    const RealOpenMM dt = integrator.getStepSize();
    const RealOpenMM dtc = dt/numDrudeSteps;      // step size for internal degrees of freedom
    const RealOpenMM dtc2 = dtc/2.0;
    const RealOpenMM dtc4 = dtc/4.0;
    const RealOpenMM dtc8 = dtc/8.0;

    // compute kinetic energies for each degrees of freedom
    // computeNHKineticEnergy(context);
    double realKE = 0.0;
    double drudeKE = 0.0;
    double centerKE = 0.0;

    // Add kinetic energy of ordinary particles.
    for (int i = 0; i < (int) normalParticles.size(); i++) {
        int index = normalParticles[i];
        if (particleInvMass[index] != 0) {
            realKE += (vel[index].dot(vel[index]))/particleInvMass[index];
        }
    }

    // Add kinetic energy of Drude particle pairs.
    for (int i = 0; i < (int) pairParticles.size(); i++) {
        int p1 = pairParticles[i].first;
        int p2 = pairParticles[i].second;
        RealOpenMM mass1fract = pairInvTotalMass[i]/particleInvMass[p1];
        RealOpenMM mass2fract = pairInvTotalMass[i]/particleInvMass[p2];
        RealVec cmVel = vel[p1]*mass1fract+vel[p2]*mass2fract;
        RealVec relVel = vel[p2]-vel[p1];
        if (pairIsCenterParticle[i])
            centerKE += (cmVel.dot(cmVel))/pairInvTotalMass[i];
        else
            realKE += (cmVel.dot(cmVel))/pairInvTotalMass[i];
        drudeKE += (relVel.dot(relVel))/pairInvReducedMass[i];
    }

    //cout << "kineticeEnergies : " << realKE << ", " << drudeKE << "\n";
    //cout << "Before NHChain real T : " << realKE/realDof/BOLTZ << ", drude T : " << drudeKE/drudeDof/BOLTZ << "\n";
    //cout << "etaDot[0] : " << etaDot[0] << ", etaDot[1] : " << etaDot[1] << "\n";

    // Calculate scaling factor for velocities using multiple Nose-Hoover chain thermostat scheme
    RealOpenMM scaleReal = 1.0;
    RealOpenMM scaleDrude = 1.0;
    RealOpenMM scaleCenter = 1.0;
    RealOpenMM scaleCM = 1.0;
    RealOpenMM expfac = 1.0;
    etaDotDot[0] = (realKE - realNkbT) / etaMass[0];
    etaDotDot[1] = (drudeKE - drudeNkbT) / etaMass[1];
    if (centerDof > 0)
        etaDotDot[2] = (centerKE - centerNkbT) / etaMass[2];

    for (int iter = 0; iter < numDrudeSteps; iter++) {

        for (int i = idxMaxNHChains; i >= 0; i--) {
            expfac = exp(-dtc8 * etaDot[i+numTempGroup]);
            etaDot[i] *= expfac;
            etaDot[i] += etaDotDot[i] * dtc4;
            etaDot[i] *= expfac;
        }

        scaleReal *= exp(-dtc2 * etaDot[0]);
        scaleDrude *= exp(-dtc2 * etaDot[1]);
        realKE *= exp(-dtc * etaDot[0]);
        drudeKE *= exp(-dtc * etaDot[1]);
        for (int i = 0; i < iNumNHChains; i++) {
            eta[i] += dtc2 * etaDot[i];
        }

        etaDotDot[0] = (realKE - realNkbT) / etaMass[0];
        etaDotDot[1] = (drudeKE - drudeNkbT) / etaMass[1];
        if (centerDof > 0) {
            scaleCenter *= exp(-dtc2 * etaDot[2]);
            centerKE *= exp(-dtc * etaDot[2]);
            etaDotDot[2] = (centerKE - centerNkbT) / etaMass[2];
        }

        for (int i = 0; i < iNumNHChains; i++) {
            expfac = exp(-dtc8 * etaDot[i+2]);
            etaDot[i] *= expfac;
            if (i > 1) { 
                double dofkbT = (i % 2 == 0 ? realkbT : drudekbT);
                etaDotDot[i] = (etaMass[i-2] * etaDot[i-2] * etaDot[i-2] - dofkbT) / etaMass[i];
            }
            etaDot[i] += etaDotDot[i] * dtc4;
            etaDot[i] *= expfac;
        }
    }
    State state = context.getOwner().getState(State::Velocities);
    vector<Vec3> veltemp = state.getVelocities();
    //state = context.getOwner().getState(State::Forces);
    //vector<Vec3> force = state.getForces();
    //cout << "After NHChain real T : " << realKE/realDof/BOLTZ << ", drude T : " << drudeKE/drudeDof/BOLTZ << "\n";
    //cout << "vscale " << scaleReal << " vscaleDrude " << scaleDrude << "\n";
    //cout << "first velocity : " << veltemp[0] << ", second velocity : " << veltemp[1] << "\n";
    //vector<RealVec>& force = extractForces(context);
    //cout << "compare to first velocity : " << vel[0] << ", second velocity : " << vel[1] << "\n";
    //cout << "first force : " << force[0] << ", second force : " << force[1] << "\n";

    // Update velocities of ordinary particles.
    for (int i = 0; i < (int) normalParticles.size(); i++) {
        int index = normalParticles[i];
        RealOpenMM invMass = particleInvMass[index];
        if (invMass != 0.0) {
            for (int j = 0; j < 3; j++)
                vel[index][j] = scaleReal*vel[index][j];
        }
    }
    
    // Update velocities of Drude particle pairs.
    for (int i = 0; i < (int) pairParticles.size(); i++) {
        int p1 = pairParticles[i].first;
        int p2 = pairParticles[i].second;
        RealOpenMM mass1fract = pairInvTotalMass[i]/particleInvMass[p1];
        RealOpenMM mass2fract = pairInvTotalMass[i]/particleInvMass[p2];
        RealVec cmVel = vel[p1]*mass1fract+vel[p2]*mass2fract;
        RealVec relVel = vel[p2]-vel[p1];
        if ( pairIsCenterParticle[i] )
            scaleCM = scaleCenter;
        else
            scaleCM = scaleReal;
        for (int j = 0; j < 3; j++) {
            cmVel[j] = scaleCM*cmVel[j];
            relVel[j] = scaleDrude*relVel[j];
        }
        vel[p1] = cmVel-relVel*mass2fract;
        vel[p2] = cmVel+relVel*mass1fract;
    }
    //state = context.getOwner().getState(State::Velocities);
    //veltemp = state.getVelocities();
    //cout << "first velocity : " << veltemp[0] << ", second velocity : " << veltemp[1] << "\n";
    //cout << "compare to first velocity : " << vel[0] << ", second velocity : " << vel[1] << "\n";
}

void ReferenceIntegrateDrudeNoseHooverStepKernel::propagateHalfVelocity(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator) {
    vector<RealVec>& vel = extractVelocities(context);
    vector<RealVec>& force = extractForces(context);

    RealOpenMM dt = integrator.getStepSize();

    // Update velocities of ordinary particles.
    for (int i = 0; i < (int) normalParticles.size(); i++) {
        int index = normalParticles[i];
        RealOpenMM invMass = particleInvMass[index];
        if (invMass != 0.0) {
            for (int j = 0; j < 3; j++)
                vel[index][j] += 0.5*dt*invMass*force[index][j];
        }
    }
    
    // Update velocities of Drude particle pairs.
    for (int i = 0; i < (int) pairParticles.size(); i++) {
        int p1 = pairParticles[i].first;
        int p2 = pairParticles[i].second;
        RealOpenMM mass1fract = pairInvTotalMass[i]/particleInvMass[p1];
        RealOpenMM mass2fract = pairInvTotalMass[i]/particleInvMass[p2];
        RealOpenMM sqrtInvTotalMass = sqrt(pairInvTotalMass[i]);
        RealOpenMM sqrtInvReducedMass = sqrt(pairInvReducedMass[i]);
        RealVec cmVel = vel[p1]*mass1fract+vel[p2]*mass2fract;
        RealVec relVel = vel[p2]-vel[p1];
        RealVec cmForce = force[p1]+force[p2];
        RealVec relForce = force[p2]*mass1fract - force[p1]*mass2fract;
        for (int j = 0; j < 3; j++) {
            cmVel[j] += 0.5*dt*pairInvTotalMass[i]*cmForce[j];
            relVel[j] += 0.5*dt*pairInvReducedMass[i]*relForce[j];
        }
        //printf("fscale : %e fscaleDrude : %e forcex : %e forcey : %e forcez : %e \n",0.5*dt,0.5*dt,force[p1][0], force[p1][1], force[p1][2]);
        vel[p1] = cmVel-relVel*mass2fract;
        vel[p2] = cmVel+relVel*mass1fract;
    }
}

double ReferenceIntegrateDrudeNoseHooverStepKernel::computeKineticEnergy(ContextImpl& context, const DrudeNoseHooverIntegrator& integrator) {
    return computeShiftedKineticEnergy(context, particleInvMass, 0.5*integrator.getStepSize());
}
