from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from drudenoseplugin import *
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
import os

# Thermostat parameters
temperature=300*kelvin
pressure = 1.0*atmospheres
barofreq = 100
REALFREQ=0.1*picosecond
DRUDEFREQ=0.1*picosecond
timestep = 0.001*picoseconds
numDrudeSteps = 20

# Initialize drude Nose Hoover integrator
integ = DrudeNoseHooverIntegrator(temperature, REALFREQ, 1*kelvin, DRUDEFREQ, timestep, numDrudeSteps)
integ.setMaxDrudeDistance(0.02)

# pdb file excluding drude particles to create modeller - equivalent to nonpolarizable simulations (typically an output from CHARMM-GUI or other preparation package)
pdb = PDBFile('nacl_1m.pdb')
# pdb file containing all postions including drude particles (typically an output from CHARMM-GUI or other preparation package)
pdb_pos = PDBFile('nacl_1m_pos.pdb')

# OpenMM assigns element based on the atom name. Fix the elements for salt ions
for i,at in enumerate(pdb.topology.atoms()):
    if at.residue.name == 'CLA':
        at.element = element.chlorine
    elif at.residue.name == 'SOD':
        at.element = element.sodium

# charmm_polar_2013.xml file comes with OpenMM installation
forcefield = ForceField('charmm_polar_2013.xml')
modeller = Modeller(pdb.topology, pdb.positions)
# add drude particles
modeller.addExtraParticles(forcefield)

system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds, rigidWater=True)
nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
customNbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == CustomNonbondedForce][0] # In charmm_polar_2013.xml, LJ interactions are saved as custom nonbonded force
drudeForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == DrudeForce][0]

# assign mass to drude particles subtracted from parent atoms (all non-hydrogen atoms)
# charmm_polar_2013.xml does not assign mass to drude particles.
for i, at in enumerate(modeller.topology.atoms()):
    if system.getParticleMass(i)/dalton > 1.1:
        system.setParticleMass(i, system.getParticleMass(i)-0.4*dalton)
    if at.name[0] == 'D':
        system.setParticleMass(i, 0.4*dalton)


barostat = MonteCarloBarostat(pressure, temperature)
system.addForce(barostat)

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}

simeq = Simulation(modeller.topology, system, integ, platform, properties)
# set positions from pdb_pos file including drude particles
simeq.context.setPositions(pdb_pos.positions)

platform =simeq.context.getPlatform()
platformname = platform.getName();
print(platformname)

print('Minimizing...')
simeq.minimizeEnergy(maxIterations=20000)
state = simeq.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
print(str(state.getKineticEnergy()))
print(str(state.getPotentialEnergy()))

position = state.getPositions()
PDBFile.writeFile(simeq.topology, position, open('min.pdb', 'w'))

print('Equilibrating...')
simeq.context.setVelocitiesToTemperature(temperature)

simeq.reporters = []
dcdfile = 'eq_npt.dcd'
logfile = 'eq_npt.log'
chkfile = 'eq_npt.chk'

simeq.reporters.append(DCDReporter(dcdfile, 1000))
simeq.reporters.append(StateDataReporter(logfile, 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, density=True,speed=True))
simeq.reporters.append(CheckpointReporter(chkfile, 10000))
simeq.reporters[1].report(simeq,state)
simeq.reporters[2].report(simeq,state)
print('Simulating...')

for i in range(1,11001):
    simeq.step(1000)

state = simeq.context.getState(getEnergy=True,getForces=True,getPositions=True, enforcePeriodicBox=True)
position = state.getPositions()
PDBFile.writeFile(simeq.topology, position, open('eq_npt.pdb', 'w'))

print('Done!')

exit()
