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
#include "raisim/World.hpp"

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

    // depth camera properties
    py::class_<raisim::DepthCamera::DepthCameraProperties>(m, "DepthCameraProperties", "Raisim depth camera properties")
        // init function?
        .def_readwrite("name", &raisim::DepthCamera::DepthCameraProperties::name)
        .def_readwrite("width", &raisim::DepthCamera::DepthCameraProperties::width)
        .def_readwrite("height", &raisim::DepthCamera::DepthCameraProperties::height)
        .def_readwrite("clipNear", &raisim::DepthCamera::DepthCameraProperties::clipNear)
        .def_readwrite("clipFar", &raisim::DepthCamera::DepthCameraProperties::clipFar)
        .def_readwrite("hFOV", &raisim::DepthCamera::DepthCameraProperties::hFOV)
        .def_readwrite("dataType", &raisim::DepthCamera::DepthCameraProperties::dataType)
        .def_readwrite("noiseType", &raisim::DepthCamera::DepthCameraProperties::noiseType)
        .def_readwrite("mean", &raisim::DepthCamera::DepthCameraProperties::mean)
        .def_readwrite("std", &raisim::DepthCamera::DepthCameraProperties::std)
    ;

    // RGB camera properties
    py::class_<raisim::RGBCamera::RGBCameraProperties>(m, "RGBCameraProperties", "Raisim RGB camera properties")
        // init function?
        .def_readwrite("name", &raisim::RGBCamera::RGBCameraProperties::name)
        .def_readwrite("width", &raisim::RGBCamera::RGBCameraProperties::width)
        .def_readwrite("height", &raisim::RGBCamera::RGBCameraProperties::height)
        .def_readwrite("clipNear", &raisim::RGBCamera::RGBCameraProperties::clipNear)
        .def_readwrite("clipFar", &raisim::RGBCamera::RGBCameraProperties::clipFar)
        .def_readwrite("hFOV", &raisim::RGBCamera::RGBCameraProperties::hFOV)
        .def_readwrite("noiseType", &raisim::RGBCamera::RGBCameraProperties::noiseType)
        .def_readwrite("mean", &raisim::RGBCamera::RGBCameraProperties::mean)
        .def_readwrite("std", &raisim::RGBCamera::RGBCameraProperties::std)
    ;

    // IMU properties
    py::class_<raisim::InertialMeasurementUnit::ImuProperties>(m, "IMUProperties", "Raisim IMU properties")
        // init function?
        .def_readwrite("name", &raisim::InertialMeasurementUnit::ImuProperties::name)
        .def_readwrite("maxAcc", &raisim::InertialMeasurementUnit::ImuProperties::maxAcc)
        .def_readwrite("maxAngVel", &raisim::InertialMeasurementUnit::ImuProperties::maxAngVel)
        .def_readwrite("noiseType", &raisim::InertialMeasurementUnit::ImuProperties::noiseType)
        .def_readwrite("mean", &raisim::InertialMeasurementUnit::ImuProperties::mean)
        .def_readwrite("std", &raisim::InertialMeasurementUnit::ImuProperties::std)
    ;

    /**********/
	/* Sensor */
	/**********/
	// Sensor class (from include/raisim/sensors/Sensors.hpp)
    // TODO: improve function documentation
    py::class_<raisim::Sensor>(m, "Sensor", "Raisim Sensor from which all other sensors inherit from.")

        // sensor init function?

        .def("getName", &raisim::Sensor::getName, R"mydelimiter(
            Get the sensor's name.

            Returns:
                str: sensor's name.
            )mydelimiter")

        .def("getType", &raisim::Sensor::getType, R"mydelimiter(get sensor type)mydelimiter" )

        .def("getPosition", [](raisim::Sensor &self){
            Vec<3> pos;
            pos = self.getPosition();
            return convert_vec_to_np(pos);
        }, R"mydelimiter(get the position of the sensor frame)mydelimiter" )

        .def("getOrientation", [](raisim::Sensor &self){
            Mat<3,3> rot;
            rot = self.getOrientation();
            return convert_mat_to_np(rot);
        }, R"mydelimiter(get orientation of the sensor frame)mydelimiter" )

        .def("getFramePosition", [](raisim::Sensor &self){
            Vec<3> pos;
            pos = self.getFramePosition();
            return convert_vec_to_np(pos);
        }, R"mydelimiter(get the position of the frame w.r.t. the nearest moving parent)mydelimiter" )

        .def("getFrameOrientation", [](raisim::Sensor &self){
            Mat<3,3> rot;
            rot = self.getFrameOrientation();
            return convert_mat_to_np(rot);
        }, R"mydelimiter(get orientation of the frame w.r.t. the nearest moving parent)mydelimiter" )

        .def("getFrameID", &raisim::Sensor::getFrameId, R"mydelimiter(get ID of frame on which the sensor is attached)mydelimiter" )

        .def("getPosInSensorFrame", [](raisim::Sensor &self){
            Vec<3> pos;
            pos = self.getPosInSensorFrame();
            return convert_vec_to_np(pos);
        }, R"mydelimiter(get the position of the frame w.r.t. the sensor frame)mydelimiter" )

        .def("getOriInSensorFrame", [](raisim::Sensor &self){
            Mat<3,3> rot;
            rot = self.getOriInSensorFrame();
            return convert_mat_to_np(rot);
        }, R"mydelimiter(get orientation of the frame w.r.t. the sensor frame)mydelimiter" )

        // make a property?
        .def("getUpdateRate", &raisim::Sensor::getUpdateRate, R"mydelimiter(get update rate in Hz)mydelimiter")
        .def("setUpdateRate", &raisim::Sensor::setUpdateRate, R"mydelimiter(set update rate in Hz)mydelimiter", py::arg("rate"))

        .def("getUpdateTimeStamp", &raisim::Sensor::getUpdateTimeStamp, R"mydelimiter(get time of last sensor measurement)mydelimiter" )
        .def("setUpdateTimeStamp", &raisim::Sensor::setUpdateTimeStamp, R"mydelimiter(set time of last sensor measurement)mydelimiter", py::arg("time"))

        .def("getMeasurementSource", &raisim::Sensor::getMeasurementSource, R"mydelimiter(get measurement source)mydelimiter")
        .def("setMeasurementSource", &raisim::Sensor::setMeasurementSource, R"mydelimiter(set measurement source)mydelimiter", py::arg("source"))

        .def("updatePose", &raisim::Sensor::updatePose, R"mydelimiter(update pose from articulated system)mydelimiter" )

        // .def("update", &raisim::Sensor::update, R"mydelimiter(update sensor measurement, using Raisim if possible)mydelimiter", py::arg("world"))

        ;


    // depth camera sensor class:
    py::class_<raisim::DepthCamera, raisim::Sensor>(m, "DepthCamera", "Raisim Depth Camera")
        // depth camera init function?

        // TODO: make output more useful? convert to numpy array?
        .def("getDepthArray", [](raisim::DepthCamera &self){
            std::vector<float> depth;
            depth = self.getDepthArray();
            py::list depth_list = py::cast(depth);
            return depth_list;
        }, R"mydelimiter(get depth data as list of floats)mydelimiter" )

        // TODO: make this faster?
        .def("get3DPoints", [](raisim::DepthCamera &self){
            std::vector< raisim::Vec<3>, AlignedAllocator<raisim::Vec<3>, 32> > threeDPoints;
            threeDPoints = self.get3DPoints();
            size_t n = threeDPoints.size();
            size_t m = threeDPoints[0].size();
            py::array_t<double> threeDPoints_array({n,m});
            for (size_t i=0; i<n; i++){
                for (size_t j=0; j<m; j++){
                    *threeDPoints_array.mutable_data(i,j) = threeDPoints[i][j];
                }
            }
            return threeDPoints_array;
        }, R"mydelimiter(get 3D points as array)mydelimiter" )

        .def("getProperties", &raisim::DepthCamera::getProperties, R"mydelimiter(get properties struct)mydelimiter" )

        .def("update", &raisim::DepthCamera::update, R"mydelimiter(update sensor measurement)mydelimiter", py::arg("world"))

        ;

    // RGB camera sensor class:
    py::class_<raisim::RGBCamera, raisim::Sensor>(m, "RGBCamera", "Raisim RGB Camera")
        // RGB camera init function?

        // TODO: make output more useful? convert to numpy array?
        .def("getImageBuffer", [](raisim::RGBCamera &self){
            std::vector<char> bgra;

            bgra = self.getImageBuffer();
            // py::list bgra_list = py::cast(bgra);
            // return bgra_list;

            size_t n = bgra.size()/4;
            size_t m = 4;
            int index = 0;
            py::array_t<double> bgra_array({n,m});
            for (size_t i=0; i<n; i++){
                for (size_t j=0; j<m; j++){
                    *bgra_array.mutable_data(i,j) = bgra[index];
                    index++;
                }
            }
            return bgra_array;
        }, R"mydelimiter(get image data as list of chars (bgra))mydelimiter" )

        .def("getProperties", &raisim::RGBCamera::getProperties, R"mydelimiter(get properties struct)mydelimiter" )

        .def("update", &raisim::RGBCamera::update, R"mydelimiter(update sensor measurement)mydelimiter", py::arg("world"))

        ;


    // IMU sensor class: (Don't need this one for now)
    py::class_<raisim::InertialMeasurementUnit, raisim::Sensor>(m, "InertialMeasurementUnit", "Raisim IMU")
        // IMU init function?

        // TODO: no valid getProperties method since prop_ is a private variable
        // // getProperties?
        // .def("getProperties", [](raisim::InertialMeasurementUnit &self){
        //     return self.prop_;
        // }, R"mydelimiter(get properties struct)mydelimiter" )

        .def("getLinearAcceleration", [](raisim::InertialMeasurementUnit &self){
            Eigen::Vector3d acc;
            acc = self.getLinearAcceleration();
            // TODO: better way to do this?
            Vec<3> acc_conv;
            acc_conv[0] = acc[0];
            acc_conv[1] = acc[1];
            acc_conv[2] = acc[2];
            return convert_vec_to_np(acc_conv);
        }, R"mydelimiter(get linear acceleration measured by sensor)mydelimiter" )

        .def("getAngularVelocity", [](raisim::InertialMeasurementUnit &self){
            Eigen::Vector3d vel;
            vel = self.getAngularVelocity();
            // TODO: better way to do this?
            Vec<3> vel_conv;
            vel_conv[0] = vel[0];
            vel_conv[1] = vel[1];
            vel_conv[2] = vel[2];
            return convert_vec_to_np(vel_conv);
        }, R"mydelimiter(get angular velocity measured by sensor)mydelimiter" )

        .def("update", &raisim::InertialMeasurementUnit::update, R"mydelimiter(update sensor measurement)mydelimiter", py::arg("world"))

        ;

}