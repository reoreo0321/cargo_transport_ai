// Persistent Gazebo Classic transport publisher for isolated paused experiments.
// Completion is checked against ROS simulation timestamps by the Python runner.
#include <gazebo/gazebo_client.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  if (!gazebo::client::setup()) return 3;
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init(argv[1]);
  auto pub=node->Advertise<gazebo::msgs::WorldControl>("~/world_control");
  pub->WaitForConnection();
  std::cout << "READY" << std::endl;
  unsigned int steps;
  while (std::cin >> steps) {
    if (steps == 0 || steps > 1000) return 4;
    gazebo::msgs::WorldControl control;
    control.set_pause(true);
    control.set_multi_step(steps);
    pub->Publish(control, true);
    std::cout << "SENT " << steps << std::endl;
  }
  gazebo::client::shutdown();
  return 0;
}
