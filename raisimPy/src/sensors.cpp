/**
 * Python wrappers for raisim.constraints using pybind11.
 *
 * Copyright (c) 2019, jhwangbo (C++), Brian Delhaisse <briandelhaisse@gmail.com> (Python wrappers)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // automatic conversion between std::vector, std::list, etc to Python list/tuples/dict
#include <pybind11/eigen.h>   // automatic conversion between Eigen data types to Numpy data types

#include "raisim/math.hpp"   // contains the definitions of Vec, Mat, etc.
#include "raisim/object/Object.hpp"
#include "raisim/helper.hpp"
#include "raisim/server/SerializationHelper.hpp"

// contains sensor information 
#include "raisim/sensors/Sensors.hpp"
#include "raisim/sensors/RGBSensor.hpp"
#include "raisim/sensors/DepthSensor.hpp"
#include "raisim/sensors/InertialMeasurementUnit.hpp"

#include "converter.hpp"  // contains code that allows to convert between the Vec, Mat to numpy arrays.

namespace py = pybind11;
using namespace raisim;


void init_sensors(py::module &m) {

    // create submodule
    py::module sensors_module = m.def_submodule("sensors", "RaiSim sensors submodule.");

    // enum for sensor types
    py::enum_<raisim::Sensor::Type>(m, "SensorType", py::arithmetic())
        .value("UNKNOWN", raisim::Sensor::Type::UNKNOWN)
        .value("RGB", raisim::Sensor::Type::RGB)
        .value("DEPTH", raisim::Sensor::Type::DEPTH)
        .value("IMU", raisim::Sensor::Type::IMU);

    // enum for measurement source
    py::enum_<raisim::Sensor::MeasurementSource>(m, "SensorMeasurementSource", py::arithmetic())
        .value("RAISIM", raisim::Sensor::MeasurementSource::RAISIM)
        .value("VISUALIZER", raisim::Sensor::MeasurementSource::VISUALIZER)
        .value("MANUAL", raisim::Sensor::MeasurementSource::MANUAL);

    // enum for frame (from depth camera)
    py::enum_<raisim::DepthCamera::Frame>(m, "DepthFrame", py::arithmetic())
        .value("SENSOR_FRAME", raisim::DepthCamera::Frame::SENSOR_FRAME)
        .value("ROOT_FRAME", raisim::DepthCamera::Frame::ROOT_FRAME)
        .value("WORLD_FRAME", raisim::DepthCamera::Frame::WORLD_FRAME);

    // enum for depth camera data type
    py::enum_<raisim::DepthCamera::DepthCameraProperties::DataType>(m, "DepthDataType", py::arithmetic())
        .value("UNKNOWN", raisim::DepthCamera::DepthCameraProperties::DataType::UNKNOWN)
        .value("COORDINATE", raisim::DepthCamera::DepthCameraProperties::DataType::COORDINATE)
        .value("DEPTH_ARRAY", raisim::DepthCamera::DepthCameraProperties::DataType::DEPTH_ARRAY);

    // enum for noise type for each sensor
    py::enum_<raisim::DepthCamera::DepthCameraProperties::NoiseType>(m, "DepthNoiseType", py::arithmetic())
        .value("GAUSSIAN", raisim::DepthCamera::DepthCameraProperties::NoiseType::GAUSSIAN)
        .value("UNIFORM", raisim::DepthCamera::DepthCameraProperties::NoiseType::UNIFORM)
        .value("NO_NOISE", raisim::DepthCamera::DepthCameraProperties::NoiseType::NO_NOISE);
    py::enum_<raisim::RGBCamera::RGBCameraProperties::NoiseType>(m, "RGBNoiseType", py::arithmetic())
        .value("GAUSSIAN", raisim::RGBCamera::RGBCameraProperties::NoiseType::GAUSSIAN)
        .value("UNIFORM", raisim::RGBCamera::RGBCameraProperties::NoiseType::UNIFORM)
        .value("NO_NOISE", raisim::RGBCamera::RGBCameraProperties::NoiseType::NO_NOISE);
    py::enum_<raisim::InertialMeasurementUnit::ImuProperties::NoiseType>(m, "IMUNoiseType", py::arithmetic())
        .value("GAUSSIAN", raisim::InertialMeasurementUnit::ImuProperties::NoiseType::GAUSSIAN)
        .value("UNIFORM", raisim::InertialMeasurementUnit::ImuProperties::NoiseType::UNIFORM)
        .value("NO_NOISE", raisim::InertialMeasurementUnit::ImuProperties::NoiseType::NO_NOISE);
    
    // sensor class:
    // sensor init function
    // sensor getPosition
    // sensor getOrientation

    /********************/
	/* SingleBodyObject */
	/********************/
	// SingleBodyObject class (from include/raisim/object/singleBodies/SingleBodyObject.hpp)
	py::class_<raisim::SingleBodyObject, raisim::Object>(m, "SingleBodyObject", "Raisim Single Object from which all single objects/bodies (such as box, sphere, etc) inherit from.")

	    .def(py::init<raisim::ObjectType>(), "Initialize the Object.", py::arg("object_type"))


        .def("getQuaternion", py::overload_cast<>(&raisim::SingleBodyObject::getQuaternion, py::const_), R"mydelimiter(
	    Get the body's orientation (expressed as a quaternion [w,x,y,z]) with respect to the world frame.

	    Returns:
	        np.array[float[4]]: quaternion [w,x,y,z].
	    )mydelimiter")


	    .def("getRotationMatrix", py::overload_cast<>(&raisim::SingleBodyObject::getRotationMatrix, py::const_), R"mydelimiter(
	    Get the body's orientation (expressed as a rotation matrix) with respect to the world frame.

	    Returns:
	        np.array[float[3,3]]: rotation matrix.
	    )mydelimiter")


	    .def("getPosition", py::overload_cast<>(&raisim::SingleBodyObject::getPosition, py::const_), R"mydelimiter(
	    Get the body's position with respect to the world frame.

	    Returns:
	        np.array[float[3]]: position in the world frame.
	    )mydelimiter")


	    .def("getComPosition", &raisim::SingleBodyObject::getComPosition, R"mydelimiter(
	    Get the body's center of mass position with respect to the world frame.

	    Returns:
	        np.array[float[3]]: center of mass position in the world frame.
	    )mydelimiter")


	    .def("getLinearVelocity", py::overload_cast<>(&raisim::SingleBodyObject::getLinearVelocity, py::const_), R"mydelimiter(
	    Get the body's linear velocity with respect to the world frame.

	    Returns:
	        np.array[float[3]]: linear velocity in the world frame.
	    )mydelimiter")


	    .def("getAngularVelocity", py::overload_cast<>(&raisim::SingleBodyObject::getAngularVelocity, py::const_), R"mydelimiter(
	    Get the body's angular velocity position with respect to the world frame.

	    Returns:
	        np.array[float[3]]: angular velocity in the world frame.
	    )mydelimiter")

	    .def("getKineticEnergy", &raisim::SingleBodyObject::getKineticEnergy, R"mydelimiter(
	    Get the body's kinetic energy.

	    Returns:
	        float: kinetic energy.
	    )mydelimiter")


	    .def("getPotentialEnergy", [](raisim::SingleBodyObject &self, py::array_t<double> gravity) {
            Vec<3> g = convert_np_to_vec<3>(gravity);
	        return self.getPotentialEnergy(g);
	    }, R"mydelimiter(
	    Get the body's potential energy due to gravity.

	    Args:
	        gravity (np.array[float[3]]): gravity vector.

	    Returns:
	        float: potential energy.
	    )mydelimiter",
	    py::arg("gravity"))


	    .def("getEnergy", [](raisim::SingleBodyObject &self, py::array_t<double> gravity) {
            Vec<3> g = convert_np_to_vec<3>(gravity);
	        return self.getEnergy(g);
	    }, R"mydelimiter(
	    Get the body's total energy.

	    Args:
	        gravity (np.array[float[3]]): gravity vector.

	    Returns:
	        float: total energy.
	    )mydelimiter",
	    py::arg("gravity"))


    // sensor class methods:
    // sensor getFramePosition
    // sensor getFrameOrientation
    // sensor getPosInSensorFrame
    // sensor getOriInSensorFrame
    // sensor getName
    // sensor getType
    // sensor getUpdateRate
    // sensor getUpdateTimeStamp
    // sensor setUpdateRate
    // sensor setUpdateTimeStamp
    // sensor updatePose
    // sensor getMeasurementSource
    // sensor setMeasurementSource
    // sensor update
    // sensor getFrameID

    // depth camera sensor class:     
    // struct for depth camera properties
    // depth camera init function
    // depth camera getDepthArray
    // depth camera get3DPoints
    // depth camera getProperties

    // RGB camera sensor class:
    // struct for RGB camera properties
    // RGB camera init function 
    // RGB camera getProperties
    // RGB camera getImageBuffer

    // IMU sensor class: (Don't need this one for now)
    // struct for IMU properties
    // IMU init function
    // IMU getLinearAcceleration
    // IMU getAngularVelocity




}