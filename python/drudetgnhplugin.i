%module drudetgnhplugin

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "OpenMMDrudeTGNH.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}

/*
 * Add units to function outputs.
*/
%pythonappend OpenMM::DrudeTGNHIntegrator::getTemperature() const %{
   val=unit.Quantity(val, unit.kelvin)
%}

%pythonappend OpenMM::DrudeTGNHIntegrator::getCouplingTime() const %{
   val=unit.Quantity(val, unit.picosecond)
%}

%pythonappend OpenMM::DrudeTGNHIntegrator::getDrudeTemperature() const %{
   val=unit.Quantity(val, unit.kelvin)
%}

%pythonappend OpenMM::DrudeTGNHIntegrator::getDrudeCouplingTime() const %{
   val=unit.Quantity(val, unit.picosecond)
%}

%pythonappend OpenMM::DrudeTGNHIntegrator::getMaxDrudeDistance() const %{
   val=unit.Quantity(val, unit.nanometer)
%}

%pythonappend OpenMM::DrudeTGNHIntegrator::getCenterParticle(int index, int& particle) const %{
%}

namespace OpenMM {

class DrudeTGNHIntegrator : public Integrator {
public:
   DrudeTGNHIntegrator(double temperature, double couplingTime, double drudeTemperature, double drudeCouplingTime, double stepSize, int drudeStepsPerRealStep=20, int numNHChains=1, int useDrudeNHChains=True, int useCOMTempGroup=True) ;

   double getTemperature() const ;
   void setTemperature(double temp) ;
   double getCouplingTime() const ;
   void setCouplingTime(double tau) ;
   double getDrudeTemperature() const ;
   void setDrudeTemperature(double temp) ;
   double getDrudeCouplingTime() const ;
   void setDrudeCouplingTime(double tau) ;
   double getMaxDrudeDistance() const ;
   void setMaxDrudeDistance(double distance) ;
   virtual void step(int steps) ;
   int getDrudeStepsPerRealStep() const ;
   void setDrudeStepsPerRealStep(int drudeSteps) ;
   int getNumNHChains() const ;
   void setNumNHChains(int numChains) ;
   int getUseDrudeNHChains() const ;
   void setUseDrudeNHChains(int useDrudeNHChains) ;
   int getUseCOMTempGroup() const ;
   void setUseCOMTempGroup(int useCOMTempGroup) ;
   int getNumTempGroups() const ;
   int addTempGroup() ;
   int addParticleTempGroup(int tempGroup) ;
   void setParticleTempGroup(int particle, int tempGroup) ;

   %apply int& OUTPUT {int& tempGroup};
   void getParticleTempGroup(int particle, int& tempGroup) const;
   %clear int& tempGroup;

};

}
