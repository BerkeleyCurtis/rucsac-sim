from mujoco_base import MuJoCoBase
from examples.control_unicycle import ControlUnicycle


def main():
    xml_path = "./xml/unicycle.xml"
    sim = ControlUnicycle(xml_path)
    sim.reset()
    sim.simulate()


if __name__ == "__main__":
    main()
