#ifndef OPENMM_DRUDENOSEHOOVERINTEGRATOR_H_
#define OPENMM_DRUDENOSEHOOVERINTEGRATOR_H_

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

#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include "openmm/internal/windowsExportDrude.h"

namespace OpenMM {

/**
 * This Integrator simulates systems that include Drude particles.  It applies two different NoseHoover
 * thermostats to different parts of the system.  The first is applied to ordinary particles (ones that
 * are not part of a Drude particle pair), as well as to the center of mass of each Drude particle pair.
 * A second thermostat, typically with a much lower temperature, is applied to the relative internal
 * displacement of each pair.
 *
 * This integrator can optionally set an upper limit on how far any Drude particle is ever allowed to
 * get from its parent particle.  This can sometimes help to improve stability.  The limit is enforced
 * with a hard wall constraint.
 * 
 * This Integrator requires the System to include a DrudeForce, which it uses to identify the Drude
 * particles.
 */

class OPENMM_EXPORT_DRUDE DrudeNoseHooverIntegrator : public Integrator {
public:
    /**
     * Create a DrudeNoseHooverIntegrator.
     *
     * @param temperature    the temperature of the main heat bath (in Kelvin)
     * @param couplingTime  the characteristic time with which couples the system to the main heat bath (in picoseconds)
     * @param drudeTemperature    the temperature of the heat bath applied to internal coordinates of Drude particles (in Kelvin)
     * @param drudeCouplingTime  the characteristic time with which couples the system to the heat bath applied to internal coordinates of Drude particles (in picoseconds)
     * @param stepSize       the step size with which to integrator the system (in picoseconds)
     * @param drudeStepsPerRealStep the number of steps with which to integrator the drude particles per single step (integer)
     * @param numNHChains    the number of Nose-Hoover chains (integer)
     * @param useDrudeNHChains    whether to use the NH-Chain for the drude d.o.f. (bool)
     */
    DrudeNoseHooverIntegrator(double temperature, double couplingTime, double drudeTemperature, double drudeCouplingTime, double stepSize, int drudeStepsPerRealStep=20, int numNHChains=1, bool useDrudeNHChains=true);
    /**
     * Get the temperature of the main heat bath (in Kelvin).
     *
     * @return the temperature of the heat bath, measured in Kelvin
     */
    double getTemperature() const {
        return temperature;
    }
    /**
     * Set the temperature of the main heat bath (in Kelvin).
     *
     * @param temp    the temperature of the heat bath, measured in Kelvin
     */
    void setTemperature(double temp) {
        temperature = temp;
    }
    /**
     * Get the coupling time t which determines how quickly the system is coupled to
     * the main heat bath (in ps).
     *
     * @return the coupling time, measured in ps
     */
    double getCouplingTime() const {
        return couplingTime;
    }
    /**
     * Set the coupling time which determines how quickly the system is coupled to
     * the main heat bath (in ps).
     *
     * @param tau    the coupling time, measured in ps
     */
    void setCouplingTime(double tau) {
        couplingTime = tau;
    }
    /**
     * Get the temperature of the heat bath applied to internal coordinates of Drude particles (in Kelvin).
     *
     * @return the temperature of the heat bath, measured in Kelvin
     */
    double getDrudeTemperature() const {
        return drudeTemperature;
    }
    /**
     * Set the temperature of the heat bath applied to internal coordinates of Drude particles (in Kelvin).
     *
     * @param temp    the temperature of the heat bath, measured in Kelvin
     */
    void setDrudeTemperature(double temp) {
        drudeTemperature = temp;
    }
    /**
     * Get the coupling time which determines how quickly the internal coordinates of Drude particles
     * are coupled to the heat bath (in ps).
     *
     * @return the coupling time, measured in ps
     */
    double getDrudeCouplingTime() const {
        return drudeCouplingTime;
    }
    /**
     * Set the coupling time which determines how quickly the internal coordinates of Drude particles
     * are coupled to the heat bath (in ps).
     *
     * @param tau    the coupling time, measured in ps
     */
    void setDrudeCouplingTime(double tau) {
        drudeCouplingTime = tau;
    }
    /**
     * Get the maximum distance a Drude particle can ever move from its parent particle, measured in nm.  This is implemented
     * with a hard wall constraint.  If this distance is set to 0 (the default), the hard wall constraint is omitted.
     */
    double getMaxDrudeDistance() const;
    /**
     * Set the maximum distance a Drude particle can ever move from its parent particle, measured in nm.  This is implemented
     * with a hard wall constraint.  If this distance is set to 0 (the default), the hard wall constraint is omitted.
     */
    void setMaxDrudeDistance(double distance);
    /**
     * Advance a simulation through time by taking a series of time steps.
     *
     * @param steps   the number of time steps to take
     */
    void step(int steps);
    /**
     * Get the number of drude steps with with to integrate per each real step (integer)
     *
     * @return the number of drude steps with with to integrate per each real step
     */
    int getDrudeStepsPerRealStep() const {
        return drudeStepsPerRealStep;
    }
    /**
     * Set the number of drude steps with with to integrate per each real step (integer)
     *
     * @param drudeSteps    the number of drude steps with with to integrate per each real step
     */
    void setDrudeStepsPerRealStep(int drudeSteps) {
        drudeStepsPerRealStep = drudeSteps;
    }
    /**
     * Get the number of Nose-Hoover chains (integer)
     *
     * @return the number of Nose-Hoover chains
     */
    int getNumNHChains() const {
        return numNHChains;
    }
    /**
     * Set the number of Nose-Hoover chains (integer)
     *
     * @param numChains    the number of Nose-Hoover chains
     */
    void setNumNHChains(int numChains) {
        numNHChains = numChains;
    }
    /**
     * Get whether to use Nose-Hoover chains or not (integer)
     *
     * @return whether to use the Nose-Hoover chains
     */
    int getUseDrudeNHChains() const {
        return useDrudeNHChains;
    }
    /**
     * Set the number of Nose-Hoover chains (integer)
     *
     * @param numChains    the number of Nose-Hoover chains
     */
    void setUseDrudeNHChains(int numChains) {
        useDrudeNHChains = numChains;
    }
    /**
     * Get the number of center particles for which independent temperature group is used.
     */
    int getNumCenterParticles() const {
        return centerParticles.size();
    }
    /**
     * Add a center particle to which independent temperature group shoul be applied.
     *
     * @param particle        the index within the System of the center particle
     * @return the index of the particle that was added
     */
    int addCenterParticle(int particle);
    /**
     * Get the parameters for a Drude particle.
     *
     * @param index                the index of the center particle for which to get parameters
     * @param[out] particle        the index within the System of the center particle
     */
    void getCenterParticle(int index, int& particle) const;
protected:
    /**
     * This will be called by the Context when it is created.  It informs the Integrator
     * of what context it will be integrating, and gives it a chance to do any necessary initialization.
     * It will also get called again if the application calls reinitialize() on the Context.
     */
    void initialize(ContextImpl& context);
    /**
     * This will be called by the Context when it is destroyed to let the Integrator do any necessary
     * cleanup.  It will also get called again if the application calls reinitialize() on the Context.
     */
    void cleanup();
    /**
     * When the user modifies the state, we need to mark that the forces need to be recalculated.
     */
    void stateChanged(State::DataType changed);
    /**
     * Get the names of all Kernels used by this Integrator.
     */
    std::vector<std::string> getKernelNames();
    /**
     * Compute the kinetic energy of the system at the current time.
     */
    double computeKineticEnergy();
private:
    double temperature, couplingTime, drudeTemperature, drudeCouplingTime, maxDrudeDistance;
    int drudeStepsPerRealStep, numNHChains;
    bool useDrudeNHChains;
    std::vector<int> centerParticles;
    Kernel kernel;
};

} // namespace OpenMM

#endif /*OPENMM_DRUDENOSEHOOVERINTEGRATOR_H_*/
