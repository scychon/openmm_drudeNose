/* -------------------------------------------------------------------------- *
 *                                OpenMMDrude                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013 Stanford University and the Authors.           *
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

#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/DrudeForce.h"
#include "openmm/DrudeTGNHIntegrator.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace OpenMM;
using namespace std;

extern "C" void registerDrudeTGNHSerializationProxies();

void testSerialization() {
    // Create an Integrator.

    DrudeTGNHIntegrator integ1(301.1, 0.1, 10.5, 0.005, 0.001);

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<DrudeTGNHIntegrator>(&integ1, "Integrator", buffer);
    DrudeTGNHIntegrator* copy = XmlSerializer::deserialize<DrudeTGNHIntegrator>(buffer);

    // Compare the two integrators to see if they are identical.
    
    DrudeTGNHIntegrator& integ2 = *copy;
    ASSERT_EQUAL(integ1.getTemperature(), integ2.getTemperature());
    ASSERT_EQUAL(integ1.getCouplingTime(), integ2.getCouplingTime());
    ASSERT_EQUAL(integ1.getDrudeTemperature(), integ2.getDrudeTemperature());
    ASSERT_EQUAL(integ1.getDrudeCouplingTime(), integ2.getDrudeCouplingTime());
    ASSERT_EQUAL(integ1.getDrudeStepsPerRealStep(), integ2.getDrudeStepsPerRealStep());
    ASSERT_EQUAL(integ1.getNumNHChains(), integ2.getNumNHChains());
    ASSERT_EQUAL(integ1.getUseDrudeNHChains(), integ2.getUseDrudeNHChains());
    ASSERT_EQUAL(integ1.getConstraintTolerance(), integ2.getConstraintTolerance());
}

int main() {
    try {
        registerDrudeTGNHSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}

