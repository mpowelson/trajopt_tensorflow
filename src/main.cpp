#include <pybind11/embed.h> // everything needed for embedding
namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::print("Starting!"); // use the Python API

    // TODO: Figure out something with the Python Path. I now have to add it manually

    py::module tensorflow = py::module::import("tensorflow");
    py::module costs = py::module::import(("trajopt_tensorflow_python"));

    costs.attr("setupProblem")(10,6);
    // Make joint velocity cost and add to problem
    py::object joint_vel_cost = costs.attr("jointVelocityCost")();
    py::object problem = costs.attr("TensorflowProblem")(joint_vel_cost);

    // Set endpoints and add to problem
    py::object start_cost = costs.attr("fixStartCost")();
    problem.attr("addCost")(start_cost);
    py::object end_cost = costs.attr("fixEndCost")();
    problem.attr("addCost")(end_cost);

    // Solve the problem
    problem.attr("solveProblem")();




}

