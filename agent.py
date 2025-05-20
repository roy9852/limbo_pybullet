import pybullet as p
import math
from typing import List
import time
import numpy as np
import openai
import base64
from io import BytesIO
from PIL import Image
import os
import re

from config import config
from simulation import Simulation
from utils import parse_answer


class Agent:
    def __init__(self):
        self.goal = None
        self.target = {
            'position': config.robot.initial_position,
            'vector': config.robot.initial_vector,
            'twist': config.robot.initial_twist,
            'close': config.robot.initial_close,
            'viewpoint': config.robot.initial_position
        }
        self.history = None
        self.agent_on = False
        self.agent_busy = False

        # Set OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    def temp(self, simulation: Simulation):
        print("[Agent] Start thinking")

        if self.goal == None:
            print("[Agent] No goal")
            return None
        
        self.target['position'] = [0.039, 0.2482, 0.099]
        self.target['vector'] = [-0.3986, -0.1649, -0.9022]
        self.target['twist'] = math.pi/2

        print("[Agent] End thinking")
        return None


    def think(self, simulation: Simulation):

        # Update target viewpoint
        self.target['viewpoint'] = simulation.robot.ee_pose[0]

        if self.goal == None:
            return None
        
        # Ground LLM
        print("[Agent] Start thinking")
        rgb_image, depth_image = simulation.camera.get_rgbd_image()
        hand_position = simulation.robot.ee_pose[0]
        banana_position = p.getBasePositionAndOrientation(simulation.object_id_1)[0]
        apple_position = p.getBasePositionAndOrientation(simulation.object_id_2)[0]

        # Prepare the prompt for the LLM
        prompt_1 = f"""
        You are a robot agent.
        You have one robot arm with two finger on your hand.
        You have a goal to achieve.
        
        You should move your hand to achieve the goal by setting the hand position.
        Hand position is given as [x, y, z] in meters.
        Usually positive x means right direction of image which is not absolute rule.
        Usually positive y means inner direction of image which is not absolute rule.
        Usually positive z means upper direction of image which is not absolute rule.
        You can refer this direction instruction, but prior the input image.

        Close is a boolean value.
        If close is True, it means the hand is closed.
        If close is False, it means the hand is open.

        You have multiple opportunities to move the hand. 
        So you do not have to move the hand in one time.
        Because repeat move-and-check, it is good to move slowly rather than fastly.

        Be careful that the hand should not collide with the objects. 
        You should be careful about collision with the target object itself.
        Keep in mind that object has size, not a one point.

        Given:
        1. Goal: {self.goal}
        2. apple center position: {apple_position}
        3. banana center position: {banana_position}
        4. Current hand position: {hand_position}
        5. Current image:
        """

        base64_img = self.img_to_base64(rgb_image)

        prompt_2 = f"""
        Think step by step. And show me your thought process.


        Your final answer should be in the following format. x, y, z is number and close is boolean: 
        '''python
        answer = [x, y, z, close]
        '''

        If goal is not proper, return original hand position.
        """

        # Build content block with optional history image
        content = [{"type": "text", "text": prompt_1}]
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}})
        content.append({"type": "text", "text": prompt_2})

        # Call the LLM
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            max_tokens=1500,
            temperature=0.0
        )

        response_text = response.choices[0].message.content
        # print(f"[Agent] LLM response: {response_text}")

        # Parse the response to get target position
        answer = parse_answer(response_text)
        if answer == "error":
            print("[Agent] Error parsing answer")
            return None
        else:
            self.target['position'] = answer[:3]
            self.target['close'] = answer[3]
            self.target['viewpoint'] = answer[:3]
            print(f"[Agent] Target position: {self.target['position']}")

        # Reset goal
        self.goal = None
        
        # Print end of thinking
        print("[Agent] End thinking")   
        return None


    def img_to_base64(self, img: np.ndarray) -> str:
        import numpy as np
        from PIL import Image
        from io import BytesIO
        import base64

        # Ensure dtype is uint8
        if img.dtype != np.uint8:
            # If float (e.g., [0, 1] range), scale to 255
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)

        # Ensure the image has 3 channels (RGB) or is grayscale
        if img.ndim == 2:
            mode = 'L'  # grayscale
        elif img.ndim == 3:
            if img.shape[2] == 3:
                mode = 'RGB'
            elif img.shape[2] == 4:
                mode = 'RGBA'
            else:
                raise ValueError(f"Unsupported channel count: {img.shape[2]}")
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        pil_image = Image.fromarray(img, mode=mode)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

