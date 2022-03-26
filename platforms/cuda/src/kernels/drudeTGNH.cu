/**
 * Calculate the kinetic energies of each degree of freedom.
 */

extern "C" __global__ void computeDrudeTGNHKineticEnergies(mixed4* __restrict__ velm, 
        const int* __restrict__ normalParticles, const int2* __restrict__ pairParticles,
        mixed* __restrict__ normalKE, mixed* __restrict__ realKE, mixed* __restrict__ drudeKE) {
 
    // Add kinetic energy of ordinary particles.
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_NORMAL_PARTICLES; i += blockDim.x*gridDim.x) {
        int index = normalParticles[i];
        mixed4 velocity = velm[index];
        if (velocity.w != 0) {
            normalKE[i] = (velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z)/velocity.w;
        }
        else
            normalKE[i] = 0;
    }
    
    // Add kinetic energy of Drude particle pairs.
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PAIRS; i += blockDim.x*gridDim.x) {
        int2 particles = pairParticles[i];
        mixed4 velocity1 = velm[particles.x];
        mixed4 velocity2 = velm[particles.y];
        mixed mass1 = RECIP(velocity1.w);
        mixed mass2 = RECIP(velocity2.w);
        mixed invTotalMass = RECIP(mass1+mass2);
        mixed invReducedMass = (mass1+mass2)*velocity1.w*velocity2.w;
        mixed mass1fract = invTotalMass*mass1;
        mixed mass2fract = invTotalMass*mass2;
        mixed4 cmVel = velocity1*mass1fract+velocity2*mass2fract;
        mixed4 relVel = velocity2-velocity1;

        realKE[i] = (cmVel.x*cmVel.x + cmVel.y*cmVel.y + cmVel.z*cmVel.z)*(mass1+mass2);
        drudeKE[i] = (relVel.x*relVel.x + relVel.y*relVel.y + relVel.z*relVel.z)*RECIP(invReducedMass);
    }
}

extern "C" __global__ void sumDrudeKineticEnergies(mixed* __restrict__ normalKE, mixed* __restrict__ realKE,
        mixed* __restrict__ drudeKE, mixed* __restrict__ kineticEnergies) {

    __shared__ mixed normalKESum[WORK_GROUP_SIZE];
    __shared__ mixed realKESum[WORK_GROUP_SIZE];
    __shared__ mixed drudeKESum[WORK_GROUP_SIZE];

    unsigned int tid = threadIdx.x;

    normalKESum[tid] = 0;
    realKESum[tid] = 0;
    drudeKESum[tid] = 0;

    for (unsigned int index = tid; index < NUM_NORMAL_PARTICLES; index += blockDim.x) {
        normalKESum[tid] += normalKE[index];
    }


    for (unsigned int index = tid; index < NUM_PAIRS; index += blockDim.x) {
        realKESum[tid] += realKE[index];
        drudeKESum[tid] += drudeKE[index];
    }

    __syncthreads();
    for (int i = WORK_GROUP_SIZE/2; i>0; i>>=1) {
        if (tid < i) {
            normalKESum[tid] += normalKESum[tid + i];
            realKESum[tid] += realKESum[tid + i];
            drudeKESum[tid] += drudeKESum[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        kineticEnergies[0] = normalKESum[0] + realKESum[0];
        kineticEnergies[1] = drudeKESum[0];
    }
}

/**
 * Calculate the center of mass velocities of each residues and get relative velocities of each particles
 */

extern "C" __global__ void calcCOMVelocities(const mixed4* __restrict__ velm,
        const int2* __restrict__ particlesInResidues, mixed4* __restrict__ comVelm, bool useCOMTempGroup) {

    // Get COM velocities
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_RESIDUES; i += blockDim.x*gridDim.x) {
        comVelm[i] = make_mixed4(0,0,0,0);
        if (useCOMTempGroup) {
        mixed comMass = 0.0;
        for (int j = 0; j < particlesInResidues[i].x; j++) {
            int index = particlesInResidues[i].y + j;
            mixed4 velocity = velm[index];
            if (velocity.w != 0) {
                mixed mass = RECIP(velocity.w);
                comVelm[i].x += velocity.x * mass;
                comVelm[i].y += velocity.y * mass;
                comVelm[i].z += velocity.z * mass;
                comMass += mass;
            }
        }
        comVelm[i].w = RECIP(comMass);
        comVelm[i].x *= comVelm[i].w;
        comVelm[i].y *= comVelm[i].w;
        comVelm[i].z *= comVelm[i].w;
        }
        else {
            comVelm[i].w = 1.0;
        }
        //if (i==0)
        //    printf("residue %d has %d particles and starts at %d and vel %f,%f,%f and mass is %f \n",i,particlesInResidues[i].x,particlesInResidues[i].y, comVelm[i].x,comVelm[i].y,comVelm[i].z, RECIP(comVelm[i].w));
    }

}

/**
 * Calculate the center of mass velocities of each residues and get relative velocities of each particles
 */

extern "C" __global__ void normalizeVelocities(const mixed4* __restrict__ velm, const int* __restrict__ particleResId,
        const mixed4* __restrict__ comVelm, mixed4* __restrict__ normVelm) {

    // Get Normalized velocities
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x) {
        normVelm[i] = make_mixed4(0,0,0,0);
        int resid = particleResId[i];
        normVelm[i].x = velm[i].x - comVelm[resid].x;
        normVelm[i].y = velm[i].y - comVelm[resid].y;
        normVelm[i].z = velm[i].z - comVelm[resid].z;
        normVelm[i].w = velm[i].w;
        //if (i==0)
        //    printf("Particle : %d, Norm Velocity : %f, velocity : %f, comVel : %f, mass : %f  \n", i,normVelm[i].x, velm[i].x, comVelm[resid].x, RECIP(normVelm[i].w));
    }
}
/**
 * Calculate the kinetic energies of each degree of freedom.
 */

extern "C" __global__ void computeNormalizedKineticEnergies(const mixed4* __restrict__ comVelm,
        const mixed4* __restrict__ normVelm, const int* __restrict__ particleTempGroup,
        const int* __restrict__ normalParticles, const int2* __restrict__ pairParticles,
        double* __restrict__ kineticEnergyBuffer) {

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    for (int i=0; i < NUM_TEMP_GROUPS+2; i++)
        kineticEnergyBuffer[tid*(NUM_TEMP_GROUPS+2)+i] = 0;

    //double comKE = 0;
    //double realKE[NUM_TEMP_GROUPS] = {0};
    //double drudeKE = 0;

    // Add kinetic energy of ordinary particles.
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_RESIDUES; i += blockDim.x*gridDim.x) {
        mixed4 velocity = comVelm[i];
        kineticEnergyBuffer[tid*(NUM_TEMP_GROUPS+2)+NUM_TEMP_GROUPS] += (velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z)/velocity.w;
        //comKE += (velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z)/velocity.w;
        //printf("i %d, comKE %f, tid %u, vx %f, vy %f, vz %f, vw %f, calc %f \n",i,comKE,tid,velocity.x,velocity.y,velocity.z,velocity.w,(velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z)/velocity.w);
        
    }

    // Add kinetic energy of ordinary particles.
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_NORMAL_PARTICLES; i += blockDim.x*gridDim.x) {
        int index = normalParticles[i];
        mixed4 velocity = normVelm[index];
        if (velocity.w != 0) {
            kineticEnergyBuffer[tid*(NUM_TEMP_GROUPS+2)+particleTempGroup[index]] += (velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z)/velocity.w;
        //    realKE[particleTempGroup[index]] += (velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z)/velocity.w;
        }
    }

    // Add kinetic energy of Drude particle pairs.
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PAIRS; i += blockDim.x*gridDim.x) {
        int2 particles = pairParticles[i];
        mixed4 velocity1 = normVelm[particles.x];
        mixed4 velocity2 = normVelm[particles.y];
        mixed mass1 = RECIP(velocity1.w);
        mixed mass2 = RECIP(velocity2.w);
        mixed invTotalMass = RECIP(mass1+mass2);
        mixed invReducedMass = (mass1+mass2)*velocity1.w*velocity2.w;
        mixed mass1fract = invTotalMass*mass1;
        mixed mass2fract = invTotalMass*mass2;
        mixed4 cmVel = velocity1*mass1fract+velocity2*mass2fract;
        mixed4 relVel = velocity2-velocity1;

        kineticEnergyBuffer[tid*(NUM_TEMP_GROUPS+2)+particleTempGroup[particles.x]] += (cmVel.x*cmVel.x + cmVel.y*cmVel.y + cmVel.z*cmVel.z)*(mass1+mass2);
        kineticEnergyBuffer[tid*(NUM_TEMP_GROUPS+2)+NUM_TEMP_GROUPS+1] += (relVel.x*relVel.x + relVel.y*relVel.y + relVel.z*relVel.z)*RECIP(invReducedMass);
        //realKE[particleTempGroup[particles.x]] += (cmVel.x*cmVel.x + cmVel.y*cmVel.y + cmVel.z*cmVel.z)*(mass1+mass2);
        //drudeKE += (relVel.x*relVel.x + relVel.y*relVel.y + relVel.z*relVel.z)*RECIP(invReducedMass);
    }
    __syncthreads();

//    for (int i=0; i < NUM_TEMP_GROUPS; i++)
//        kineticEnergyBuffer[tid*(NUM_TEMP_GROUPS+2)+i] = realKE[i];
//    kineticEnergyBuffer[tid*(NUM_TEMP_GROUPS+2)+NUM_TEMP_GROUPS] = comKE;
//    kineticEnergyBuffer[tid*(NUM_TEMP_GROUPS+2)+NUM_TEMP_GROUPS+1] = drudeKE;
//    __syncthreads();
//    if (tid==0) {
//        printf("comKE %f, realKE %f, drude %f tid %u \n",comKE,realKE[0],drudeKE,tid);
//        printf("blockdim %d griddim %d num_temp_group %d buffersize %d comKE %f, realKE %f, drude %f tid %u \n",blockDim.x,gridDim.x,NUM_TEMP_GROUPS+2,int(*(&kineticEnergyBuffer+1)-kineticEnergyBuffer),comKE,realKE[0],drudeKE,tid);
//    }
}

extern "C" __global__ void sumNormalizedKineticEnergies(double* __restrict__ kineticEnergyBuffer, double* __restrict__ kineticEnergies, int bufferSize) {
    // Sum the threads in this group.
    __shared__ double temp[WORK_GROUP_SIZE*(NUM_TEMP_GROUPS+2)];
    unsigned int tid = threadIdx.x;

    for (unsigned int i=0; i < NUM_TEMP_GROUPS+2; i++)
        temp[WORK_GROUP_SIZE*i+tid] = 0;
    __syncthreads();

    for (unsigned int index = tid*(NUM_TEMP_GROUPS+2); index < bufferSize; index += blockDim.x*(NUM_TEMP_GROUPS+2)) {
        for (unsigned int i=0; i < NUM_TEMP_GROUPS+2; i++)
            temp[WORK_GROUP_SIZE*i+tid] += kineticEnergyBuffer[index + i];
    }
    __syncthreads();
    if (tid < 32) {
        for (unsigned int i=0; i < NUM_TEMP_GROUPS+2; i+=1) temp[WORK_GROUP_SIZE*i+tid] += temp[WORK_GROUP_SIZE*i+tid+32];
        __syncthreads();
        if (tid < 16) {
            for (unsigned int i=0; i < NUM_TEMP_GROUPS+2; i+=1) temp[WORK_GROUP_SIZE*i+tid] += temp[WORK_GROUP_SIZE*i+tid+16];
        }
        __syncthreads();
        if (tid < 8) {
            for (unsigned int i=0; i < NUM_TEMP_GROUPS+2; i+=1) temp[WORK_GROUP_SIZE*i+tid] += temp[WORK_GROUP_SIZE*i+tid+8];
        }
        __syncthreads();
        if (tid < 4) {
            for (unsigned int i=0; i < NUM_TEMP_GROUPS+2; i+=1) temp[WORK_GROUP_SIZE*i+tid] += temp[WORK_GROUP_SIZE*i+tid+4];
        }
        __syncthreads();
        if (tid < 2) {
            for (unsigned int i=0; i < NUM_TEMP_GROUPS+2; i+=1) temp[WORK_GROUP_SIZE*i+tid] += temp[WORK_GROUP_SIZE*i+tid+2];
        }
        __syncthreads();
    }
    __syncthreads();
    if (tid == 0) {
        for (unsigned int i=0; i < NUM_TEMP_GROUPS+2; i++) {
            kineticEnergies[i] = temp[WORK_GROUP_SIZE*i]+temp[WORK_GROUP_SIZE*i+1];
        }
    }
}


/**
 * Perform the velocity update of TGNH Chain integration.
 */

extern "C" __global__ void integrateDrudeTGNHChain(mixed4* __restrict__ velm, const mixed4* __restrict__ normVelm,
        const int* __restrict__ normalParticles, const int2* __restrict__ pairParticles, const int* __restrict__ particleTempGroup, const mixed* __restrict__ vscaleFactors) {

    mixed vscaleCOM = vscaleFactors[NUM_TEMP_GROUPS];
    mixed vscaleDrude = vscaleFactors[NUM_TEMP_GROUPS+1];
    // Update normal particles.
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_NORMAL_PARTICLES; i += blockDim.x*gridDim.x) {
        int index = normalParticles[i];
        mixed4 velocity = velm[index];
        mixed4 velRel = normVelm[index];
        mixed vscale = vscaleFactors[particleTempGroup[index]];
        if (velocity.w != 0) {
            velocity.x = vscale*velRel.x + vscaleCOM*(velocity.x-velRel.x);
            velocity.y = vscale*velRel.y + vscaleCOM*(velocity.y-velRel.y);
            velocity.z = vscale*velRel.z + vscaleCOM*(velocity.z-velRel.z);
            velm[index] = velocity;
        }
    }
    
    // Update Drude particle pairs.
    
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PAIRS; i += blockDim.x*gridDim.x) {
        int2 particles = pairParticles[i];
        mixed vscaleCM = vscaleFactors[particleTempGroup[particles.x]];
        mixed4 velocity1 = velm[particles.x];
        mixed4 velocity2 = velm[particles.y];
        mixed4 velRel1 = normVelm[particles.x];
        mixed4 velRel2 = normVelm[particles.y];
        mixed4 velCOM1 = velocity1 - velRel1;
        mixed4 velCOM2 = velocity2 - velRel2;
        mixed mass1 = RECIP(velocity1.w);
        mixed mass2 = RECIP(velocity2.w);
        mixed invTotalMass = RECIP(mass1+mass2);
        mixed mass1fract = invTotalMass*mass1;
        mixed mass2fract = invTotalMass*mass2;
        mixed4 cmVel = velRel1*mass1fract+velRel2*mass2fract;
        mixed4 relVel = velRel2-velRel1;
        cmVel.x = vscaleCM*cmVel.x;
        cmVel.y = vscaleCM*cmVel.y;
        cmVel.z = vscaleCM*cmVel.z;
        relVel.x = vscaleDrude*relVel.x;
        relVel.y = vscaleDrude*relVel.y;
        relVel.z = vscaleDrude*relVel.z;
        velocity1.x = cmVel.x-relVel.x*mass2fract + vscaleCOM*velCOM1.x;
        velocity1.y = cmVel.y-relVel.y*mass2fract + vscaleCOM*velCOM1.y;
        velocity1.z = cmVel.z-relVel.z*mass2fract + vscaleCOM*velCOM1.z;
        velocity2.x = cmVel.x+relVel.x*mass1fract + vscaleCOM*velCOM2.x;
        velocity2.y = cmVel.y+relVel.y*mass1fract + vscaleCOM*velCOM2.y;
        velocity2.z = cmVel.z+relVel.z*mass1fract + vscaleCOM*velCOM2.z;
        velm[particles.x] = velocity1;
        velm[particles.y] = velocity2;
    }
}

/**
 * Perform the velocity update of TGNH Chain integration.
 */

extern "C" __global__ void integrateDrudeTGNHVelocities(mixed4* __restrict__ velm, const long long* __restrict__ force, mixed4* __restrict__ posDelta,
        const int* __restrict__ normalParticles, const int2* __restrict__ pairParticles, const mixed2* __restrict__ dt, const mixed fscale,
        const mixed fscaleDrude, bool updatePosDelta) {
    mixed stepSize = dt[0].y;
    
    // Update normal particles.

    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_NORMAL_PARTICLES; i += blockDim.x*gridDim.x) {
        int index = normalParticles[i];
        mixed4 velocity = velm[index];
        if (velocity.w != 0) {
            velocity.x = velocity.x + fscale*velocity.w*force[index];
            velocity.y = velocity.y + fscale*velocity.w*force[index+PADDED_NUM_ATOMS];
            velocity.z = velocity.z + fscale*velocity.w*force[index+PADDED_NUM_ATOMS*2];
            velm[index] = velocity;
            if (updatePosDelta) {
                posDelta[index] = make_mixed4(stepSize*velocity.x, stepSize*velocity.y, stepSize*velocity.z, 0);
            }
        }
    }
    
    // Update Drude particle pairs.
    
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PAIRS; i += blockDim.x*gridDim.x) {
        int2 particles = pairParticles[i];
        mixed4 velocity1 = velm[particles.x];
        mixed4 velocity2 = velm[particles.y];
        mixed mass1 = RECIP(velocity1.w);
        mixed mass2 = RECIP(velocity2.w);
        mixed invTotalMass = RECIP(mass1+mass2);
        mixed invReducedMass = (mass1+mass2)*velocity1.w*velocity2.w;
        mixed mass1fract = invTotalMass*mass1;
        mixed mass2fract = invTotalMass*mass2;
        mixed4 cmVel = velocity1*mass1fract+velocity2*mass2fract;
        mixed4 relVel = velocity2-velocity1;
        mixed3 force1 = make_mixed3(force[particles.x], force[particles.x+PADDED_NUM_ATOMS], force[particles.x+PADDED_NUM_ATOMS*2]);
        mixed3 force2 = make_mixed3(force[particles.y], force[particles.y+PADDED_NUM_ATOMS], force[particles.y+PADDED_NUM_ATOMS*2]);
        mixed3 cmForce = force1+force2;
        mixed3 relForce = force2*mass1fract - force1*mass2fract;
        cmVel.x = cmVel.x + fscale*invTotalMass*cmForce.x;
        cmVel.y = cmVel.y + fscale*invTotalMass*cmForce.y;
        cmVel.z = cmVel.z + fscale*invTotalMass*cmForce.z;
        relVel.x = relVel.x + fscaleDrude*invReducedMass*relForce.x;
        relVel.y = relVel.y + fscaleDrude*invReducedMass*relForce.y;
        relVel.z = relVel.z + fscaleDrude*invReducedMass*relForce.z;
        velocity1.x = cmVel.x-relVel.x*mass2fract;
        velocity1.y = cmVel.y-relVel.y*mass2fract;
        velocity1.z = cmVel.z-relVel.z*mass2fract;
        velocity2.x = cmVel.x+relVel.x*mass1fract;
        velocity2.y = cmVel.y+relVel.y*mass1fract;
        velocity2.z = cmVel.z+relVel.z*mass1fract;
        velm[particles.x] = velocity1;
        velm[particles.y] = velocity2;
        if (updatePosDelta) {
            posDelta[particles.x] = make_mixed4(stepSize*velocity1.x, stepSize*velocity1.y, stepSize*velocity1.z, 0);
            posDelta[particles.y] = make_mixed4(stepSize*velocity2.x, stepSize*velocity2.y, stepSize*velocity2.z, 0);
        }
    }
}

/**
 * Perform the velocity update of TGNH Chain integration.
 */

extern "C" __global__ void integrateDrudeTGNHVelocitiesAndPositions(mixed4* __restrict__ velm, const long long* __restrict__ force, mixed4* __restrict__ posDelta,
        const int* __restrict__ normalParticles, const int2* __restrict__ pairParticles, const mixed2* __restrict__ dt, const mixed vscale, const mixed fscale,
        const mixed vscaleDrude, const mixed fscaleDrude, bool updatePosDelta) {
    mixed stepSize = dt[0].y;
    
    // Update normal particles.

    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_NORMAL_PARTICLES; i += blockDim.x*gridDim.x) {
        int index = normalParticles[i];
        mixed4 velocity = velm[index];
        if (velocity.w != 0) {
            velocity.x = vscale*velocity.x + fscale*velocity.w*force[index];
            velocity.y = vscale*velocity.y + fscale*velocity.w*force[index+PADDED_NUM_ATOMS];
            velocity.z = vscale*velocity.z + fscale*velocity.w*force[index+PADDED_NUM_ATOMS*2];
            velm[index] = velocity;
            if (updatePosDelta) {
                posDelta[index] = make_mixed4(stepSize*velocity.x, stepSize*velocity.y, stepSize*velocity.z, 0);
            }
        }
    }
    
    // Update Drude particle pairs.
    
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PAIRS; i += blockDim.x*gridDim.x) {
        int2 particles = pairParticles[i];
        mixed4 velocity1 = velm[particles.x];
        mixed4 velocity2 = velm[particles.y];
        mixed mass1 = RECIP(velocity1.w);
        mixed mass2 = RECIP(velocity2.w);
        mixed invTotalMass = RECIP(mass1+mass2);
        mixed invReducedMass = (mass1+mass2)*velocity1.w*velocity2.w;
        mixed mass1fract = invTotalMass*mass1;
        mixed mass2fract = invTotalMass*mass2;
        mixed4 cmVel = velocity1*mass1fract+velocity2*mass2fract;
        mixed4 relVel = velocity2-velocity1;
        mixed3 force1 = make_mixed3(force[particles.x], force[particles.x+PADDED_NUM_ATOMS], force[particles.x+PADDED_NUM_ATOMS*2]);
        mixed3 force2 = make_mixed3(force[particles.y], force[particles.y+PADDED_NUM_ATOMS], force[particles.y+PADDED_NUM_ATOMS*2]);
        mixed3 cmForce = force1+force2;
        mixed3 relForce = force2*mass1fract - force1*mass2fract;
        cmVel.x = vscale*cmVel.x + fscale*invTotalMass*cmForce.x;
        cmVel.y = vscale*cmVel.y + fscale*invTotalMass*cmForce.y;
        cmVel.z = vscale*cmVel.z + fscale*invTotalMass*cmForce.z;
        relVel.x = vscaleDrude*relVel.x + fscaleDrude*invReducedMass*relForce.x;
        relVel.y = vscaleDrude*relVel.y + fscaleDrude*invReducedMass*relForce.y;
        relVel.z = vscaleDrude*relVel.z + fscaleDrude*invReducedMass*relForce.z;
        velocity1.x = cmVel.x-relVel.x*mass2fract;
        velocity1.y = cmVel.y-relVel.y*mass2fract;
        velocity1.z = cmVel.z-relVel.z*mass2fract;
        velocity2.x = cmVel.x+relVel.x*mass1fract;
        velocity2.y = cmVel.y+relVel.y*mass1fract;
        velocity2.z = cmVel.z+relVel.z*mass1fract;
        velm[particles.x] = velocity1;
        velm[particles.y] = velocity2;
        if (updatePosDelta) {
            posDelta[particles.x] = make_mixed4(stepSize*velocity1.x, stepSize*velocity1.y, stepSize*velocity1.z, 0);
            posDelta[particles.y] = make_mixed4(stepSize*velocity2.x, stepSize*velocity2.y, stepSize*velocity2.z, 0);
        }
    }
}

/**
 * Perform the position update of TGNH integration.
 */

extern "C" __global__ void integrateDrudeTGNHPositions(real4* __restrict__ posq, real4* __restrict__ posqCorrection, const mixed4* __restrict__ posDelta, mixed4* __restrict__ velm, const mixed2* __restrict__ dt) {
    double invStepSize = 1.0/dt[0].y;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    while (index < NUM_ATOMS) {
        mixed4 vel = velm[index];
        if (vel.w != 0) {
#ifdef USE_MIXED_PRECISION
 
            real4 pos1 = posq[index];
            real4 pos2 = posqCorrection[index];
            mixed4 pos = make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
#else
            real4 pos = posq[index];
#endif
            mixed4 delta = posDelta[index];
            pos.x += delta.x;
            pos.y += delta.y;
            pos.z += delta.z;
            vel.x = (mixed) (invStepSize*delta.x);
            vel.y = (mixed) (invStepSize*delta.y);
            vel.z = (mixed) (invStepSize*delta.z);
#ifdef USE_MIXED_PRECISION
            posq[index] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) pos.w);
            posqCorrection[index] = make_real4(pos.x-(real) pos.x, pos.y-(real) pos.y, pos.z-(real) pos.z, 0);
#else
            posq[index] = pos;
#endif
            velm[index] = vel;
        }
        index += blockDim.x*gridDim.x;
    }
}

/**
 * Apply hard wall constraints
 */
extern "C" __global__ void applyHardWallConstraints(real4* __restrict__ posq, real4* __restrict__ posqCorrection, mixed4* __restrict__ velm,
        const int2* __restrict__ pairParticles, const mixed2* __restrict__ dt, mixed maxDrudeDistance, mixed hardwallscaleDrude) {
    mixed stepSize = dt[0].y;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PAIRS; i += blockDim.x*gridDim.x) {
        int2 particles = pairParticles[i];
#ifdef USE_MIXED_PRECISION
        real4 posReal1 = posq[particles.x];
        real4 posReal2 = posq[particles.y];
        real4 posCorr1 = posqCorrection[particles.x];
        real4 posCorr2 = posqCorrection[particles.y];
        mixed4 pos1 = make_mixed4(posReal1.x+(mixed)posCorr1.x, posReal1.y+(mixed)posCorr1.y, posReal1.z+(mixed)posCorr1.z, posReal1.w);
        mixed4 pos2 = make_mixed4(posReal2.x+(mixed)posCorr2.x, posReal2.y+(mixed)posCorr2.y, posReal2.z+(mixed)posCorr2.z, posReal2.w);
#else
        mixed4 pos1 = posq[particles.x];
        mixed4 pos2 = posq[particles.y];
#endif
        mixed4 delta = pos1-pos2;
        mixed r = SQRT(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
        mixed rInv = RECIP(r);
        if (rInv*maxDrudeDistance < 1) {
            // The constraint has been violated, so make the inter-particle distance "bounce"
            // off the hard wall.

            mixed4 bondDir = delta*rInv;
            mixed4 vel1 = velm[particles.x];
            mixed4 vel2 = velm[particles.y];
            mixed mass1 = RECIP(vel1.w);
            mixed mass2 = RECIP(vel2.w);
            mixed deltaR = r-maxDrudeDistance;
            mixed deltaT = stepSize;
            mixed dotvr1 = vel1.x*bondDir.x + vel1.y*bondDir.y + vel1.z*bondDir.z;
            mixed4 vb1 = bondDir*dotvr1;
            mixed4 vp1 = vel1-vb1;
            if (vel2.w == 0) {
                // The parent particle is massless, so move only the Drude particle.

                if (dotvr1 != 0)
                    deltaT = deltaR/fabs(dotvr1);
                if (deltaT > stepSize)
                    deltaT = stepSize;
                dotvr1 = -dotvr1*hardwallscaleDrude/(fabs(dotvr1)*SQRT(mass1));
                mixed dr = -deltaR + deltaT*dotvr1;
                pos1.x += bondDir.x*dr;
                pos1.y += bondDir.y*dr;
                pos1.z += bondDir.z*dr;
#ifdef USE_MIXED_PRECISION
                posq[particles.x] = make_real4((real) pos1.x, (real) pos1.y, (real) pos1.z, (real) pos1.w);
                posqCorrection[particles.x] = make_real4(pos1.x-(real) pos1.x, pos1.y-(real) pos1.y, pos1.z-(real) pos1.z, 0);
#else
                posq[particles.x] = pos1;
#endif
                vel1.x = vp1.x + bondDir.x*dotvr1;
                vel1.y = vp1.y + bondDir.y*dotvr1;
                vel1.z = vp1.z + bondDir.z*dotvr1;
                velm[particles.x] = vel1;
            }
            else {
                // Move both particles.

                mixed invTotalMass = RECIP(mass1+mass2);
                mixed dotvr2 = vel2.x*bondDir.x + vel2.y*bondDir.y + vel2.z*bondDir.z;
                mixed4 vb2 = bondDir*dotvr2;
                mixed4 vp2 = vel2-vb2;
                mixed vbCMass = (mass1*dotvr1 + mass2*dotvr2)*invTotalMass;
                dotvr1 -= vbCMass;
                dotvr2 -= vbCMass;
                if (dotvr1 != dotvr2)
                    deltaT = deltaR/fabs(dotvr1-dotvr2);
                if (deltaT > stepSize)
                    deltaT = stepSize;
                mixed vBond = hardwallscaleDrude/SQRT(mass1);
                dotvr1 = -dotvr1*vBond*mass2*invTotalMass/fabs(dotvr1);
                dotvr2 = -dotvr2*vBond*mass1*invTotalMass/fabs(dotvr2);
                mixed dr1 = -deltaR*mass2*invTotalMass + deltaT*dotvr1;
                mixed dr2 = deltaR*mass1*invTotalMass + deltaT*dotvr2;
                dotvr1 += vbCMass;
                dotvr2 += vbCMass;
                pos1.x += bondDir.x*dr1;
                pos1.y += bondDir.y*dr1;
                pos1.z += bondDir.z*dr1;
                pos2.x += bondDir.x*dr2;
                pos2.y += bondDir.y*dr2;
                pos2.z += bondDir.z*dr2;
#ifdef USE_MIXED_PRECISION
                posq[particles.x] = make_real4((real) pos1.x, (real) pos1.y, (real) pos1.z, (real) pos1.w);
                posq[particles.y] = make_real4((real) pos2.x, (real) pos2.y, (real) pos2.z, (real) pos2.w);
                posqCorrection[particles.x] = make_real4(pos1.x-(real) pos1.x, pos1.y-(real) pos1.y, pos1.z-(real) pos1.z, 0);
                posqCorrection[particles.y] = make_real4(pos2.x-(real) pos2.x, pos2.y-(real) pos2.y, pos2.z-(real) pos2.z, 0);
#else
                posq[particles.x] = pos1;
                posq[particles.y] = pos2;
#endif
                vel1.x = vp1.x + bondDir.x*dotvr1;
                vel1.y = vp1.y + bondDir.y*dotvr1;
                vel1.z = vp1.z + bondDir.z*dotvr1;
                vel2.x = vp2.x + bondDir.x*dotvr2;
                vel2.y = vp2.y + bondDir.y*dotvr2;
                vel2.z = vp2.z + bondDir.z*dotvr2;
                velm[particles.x] = vel1;
                velm[particles.y] = vel2;
            }
        }
    }
}
