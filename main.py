import threading
import time
import math
import pybullet as p

from simulation import Simulation
from agent import Agent
from detector import Detector
from utils import save_rgbd_image


def simulation_loop(simulation: Simulation, agent: Agent, detector: Detector):
    '''
    Subscribe:
        - target of agent class
    Publish:
        - state of simulation class
    '''
    while True:
        # 1. Define subscribed variables
        target = agent.target
        
        # 2. Close/open gripper
        if target['close']:
            simulation.robot.close_gripper()
        else:
            simulation.robot.open_gripper()

        # 3. Move robot
        if simulation.robot.solve_ik(target['position'], target['vector'], target['twist']):
            simulation.robot.move()
            simulation.robot.get_ee_pose()
        else:
            print("[sim_loop] Failed to solve robot IK.")

        # 4. Move camera    
        if simulation.camera.solve_ik(target['viewpoint']):
            simulation.camera.move()
            simulation.camera.get_ee_pose()
        else:
            print("[sim_loop] Failed to solve camera IK.")

        # 5. Step simulation
        _, _ = simulation.camera.get_rgbd_image()
        simulation.step()
        time.sleep(1.0/240.0)


def agent_loop(simulation: Simulation, agent: Agent, detector: Detector):
    '''
    Subscribe:
        - something
    Publish:
        - something
    '''
    while True:
        # 1. Define subscribed variables
        #####

        # 2. Use LLM if necessary
        if agent.agent_on and not agent.agent_busy:
            agent.agent_busy = True
            # detector.condition = agent.think(simulation)
            detector.condition = agent.think(simulation)
            agent.agent_busy = False
        else:
            agent.agent_busy = False

        time.sleep(1.0)


def detector_loop(simulation: Simulation, agent: Agent, detector: Detector):
    '''
    Subscribe:
        - state of simulation class
        - condition of detector class
    Publish:
        - agent_on of agent class
    '''
    while True:
        # 1. Define subscribed variables
        #####

        # 2. Check condition
        if simulation.robot.is_target_reached():
            agent.agent_on = True
        else:
            agent.agent_on = False

        # 3. For now, the agent is always on. This would be changed in the future.
        agent.agent_on = True

        time.sleep(1.0)


def main_loop():
    # 1. Initialize all nodes
    simulation = Simulation()
    agent = Agent()
    detector = Detector()

    goal_1 = """
    Move robot hand based on the current situation.
    If robot hand is above the banana, move your hand above the apple.
    If robot hand is above the apple, move your hand between of the apple and the banana.
    If none of them, move your hand to above the banana.
    Do not move in z-direction.
    """

    goal_2 ="""
    Grasp the apple in the image.
    """

    agent.goal = None

    # 2. Define components threads
    simulation_thread = threading.Thread(target=simulation_loop, args=(simulation, agent, detector), daemon=True)
    agent_thread = threading.Thread(target=agent_loop, args=(simulation, agent, detector), daemon=True)
    detector_thread = threading.Thread(target=detector_loop, args=(simulation, agent, detector), daemon=True)

    # 3. Start components threads   
    simulation_thread.start()
    agent_thread.start()
    detector_thread.start()

    # 4. Main loop
    frame_idx = 0
    while True:
        if agent.goal == None:
            agent.goal = input("Enter the goal: ")
        # print(f"[Main] target position: [{agent.target['position'][0]:.3f}, {agent.target['position'][1]:.3f}, {agent.target['position'][2]:.3f}]")
        # print(f"[Main] current position: [{simulation.robot.ee_pose[0][0]:.3f}, {simulation.robot.ee_pose[0][1]:.3f}, {simulation.robot.ee_pose[0][2]:.3f}]")
        # print()

        color_img, depth_img = simulation.camera.get_rgbd_image()
        # save_rgbd_image(color_img, depth_img, save_dir="images", prefix="frame", frame_idx=frame_idx)

        time.sleep(1.0)
        frame_idx += 1


if __name__ == "__main__":
    main_loop()  