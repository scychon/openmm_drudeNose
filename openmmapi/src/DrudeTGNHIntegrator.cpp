/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2015 Stanford University and the Authors.      *
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

#include "openmm/DrudeTGNHIntegrator.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/DrudeTGNHKernels.h"
#include <ctime>
#include <string>
#include <iostream>
#include <typeinfo>

using namespace OpenMM;
using std::string;
using std::vector;

DrudeTGNHIntegrator::DrudeTGNHIntegrator(double temperature, double couplingTime, double drudeTemperature, double drudeCouplingTime, double stepSize, int drudeStepsPerRealStep, int numNHChains, bool useDrudeNHChains, bool useCOMTempGroup) {
    setTemperature(temperature);
    setCouplingTime(couplingTime);
    setDrudeTemperature(drudeTemperature);
    setDrudeCouplingTime(drudeCouplingTime);
    setMaxDrudeDistance(0);
    setStepSize(stepSize);
    setDrudeStepsPerRealStep(drudeStepsPerRealStep);
    setNumNHChains(numNHChains);
    setUseDrudeNHChains(useDrudeNHChains);
    setUseCOMTempGroup(useCOMTempGroup);
    setConstraintTolerance(1e-5);
}

int DrudeTGNHIntegrator::addTempGroup() {
    tempGroups.push_back(tempGroups.size());
    return tempGroups.size()-1;
}

int DrudeTGNHIntegrator::addParticleTempGroup(int tempGroup) {
    ASSERT_VALID_INDEX(tempGroup, tempGroups);
    particleTempGroup.push_back(tempGroup);
    return particleTempGroup.size()-1;
}

void DrudeTGNHIntegrator::setParticleTempGroup(int particle, int tempGroup) {
    ASSERT_VALID_INDEX(particle, particleTempGroup);
    ASSERT_VALID_INDEX(tempGroup, tempGroups);
    particleTempGroup[particle] = tempGroup;
}

void DrudeTGNHIntegrator::getParticleTempGroup(int particle, int& tempGroup) const {
    ASSERT_VALID_INDEX(particle, particleTempGroup);
    tempGroup = particleTempGroup[particle];
}

double DrudeTGNHIntegrator::getResInvMass(int resid) const {
    ASSERT_VALID_INDEX(resid, residueInvMasses);
    return residueInvMasses[resid];
}

int DrudeTGNHIntegrator::getParticleResId(int particle) const {
    ASSERT_VALID_INDEX(particle, particleResId);
    return particleResId[particle];
}

double DrudeTGNHIntegrator::getMaxDrudeDistance() const {
    return maxDrudeDistance;
}

void DrudeTGNHIntegrator::setMaxDrudeDistance(double distance) {
    if (distance < 0)
        throw OpenMMException("setMaxDrudeDistance: Distance cannot be negative");
    maxDrudeDistance = distance;
}

void DrudeTGNHIntegrator::initialize(ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMMException("This Integrator is already bound to a context");
    const DrudeForce* force = NULL;
    const System& system = contextRef.getSystem();
    isKESumValid = false;
    hasBarostat = false;
    for (int i = 0; i < system.getNumForces(); i++) {
        if (dynamic_cast<const DrudeForce*>(&system.getForce(i)) != NULL) {
            if (force == NULL)
                force = dynamic_cast<const DrudeForce*>(&system.getForce(i));
            else
                throw OpenMMException("The System contains multiple DrudeForces");
        }
        std::string str(typeid(system.getForce(i)).name());
        if (str.find("Baro") != std::string::npos) {
            std::cout << typeid(system.getForce(i)).name() << "force group name\n";
            hasBarostat = true;
        }
    }
    if (force == NULL)
        throw OpenMMException("The System does not contain a DrudeForce");

    // If particleTempGroup is not assigned, assign all to single temperature group
    if (particleTempGroup.size() == 0) {
        if (tempGroups.size() == 0)
            tempGroups.push_back(0);
        for (int i = 0; i < system.getNumParticles(); i++)
            particleTempGroup.push_back(0);
    }
    else if (particleTempGroup.size() != system.getNumParticles())
        throw OpenMMException("Number of particles assigned with temperature groups does not match the number of system particles");

    particleResId = std::vector<int>(system.getNumParticles(), -1);
    std::vector<std::vector<int> > molecules = contextRef.getMolecules();
    std::cout << "num molecules : " << (int) molecules.size() << "\n";
    int numResidues = (int) molecules.size();
    for (int i = 0; i < numResidues; i++)
        for (int j = 0; j < (int) molecules[i].size(); j++)
            particleResId[molecules[i][j]] = i;

    std::cout << "particleResId assigned ! \n";

    residueMasses = std::vector<double>(numResidues, 0.0);
    for (int i = 0; i < system.getNumParticles(); i++)
        residueMasses[particleResId[i]] += system.getParticleMass(i);

    std::cout << "residue Masses assigned ! \n";

    for (int i = 0; i < numResidues; i++)
        residueInvMasses.push_back(1.0/residueMasses[i]);
 
    std::cout << "residue inverse Masses assigned ! \n";
    context = &contextRef;
    owner = &contextRef.getOwner();
    kernel = context->getPlatform().createKernel(IntegrateDrudeTGNHStepKernel::Name(), contextRef);
    kernel.getAs<IntegrateDrudeTGNHStepKernel>().initialize(contextRef.getSystem(), *this, *force);
}

void DrudeTGNHIntegrator::cleanup() {
    kernel = Kernel();
}

void DrudeTGNHIntegrator::stateChanged(State::DataType changed) {
    isKESumValid = false;
    if (context != NULL)
        context->calcForcesAndEnergy(true, false);
}

vector<string> DrudeTGNHIntegrator::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(IntegrateDrudeTGNHStepKernel::Name());
    return names;
}

double DrudeTGNHIntegrator::computeKineticEnergy() {
    return kernel.getAs<IntegrateDrudeTGNHStepKernel>().computeKineticEnergy(*context, *this, isKESumValid);
}

void DrudeTGNHIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMMException("This Integrator is not bound to a context!");    
    for (int i = 0; i < steps; ++i) {
        if (context->updateContextState())
            context->calcForcesAndEnergy(true, false);
        else if (context->getLastForceGroups() >= 0)
            context->calcForcesAndEnergy(true, false);

        kernel.getAs<IntegrateDrudeTGNHStepKernel>().execute(*context, *this);
        isKESumValid = true;
    }
}
