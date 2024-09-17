import threading
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pickle
import os
import logging
from queue import Queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        limit = np.sqrt(6 / (input_size + hidden_size))
        self.weights_input_hidden = np.random.uniform(-limit, limit, (input_size, hidden_size))
        limit = np.sqrt(6 / (hidden_size + output_size))
        self.weights_hidden_output = np.random.uniform(-limit, limit, (hidden_size, output_size))

    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden)
        self.hidden_layer = np.maximum(0, self.hidden_input)
        self.output_input = np.dot(self.hidden_layer, self.weights_hidden_output)
        self.output_layer = self.softmax(self.output_input)
        return self.output_layer

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def train(self, x, y):
        output = self.forward(x)
        error = output - y
        delta_output = error
        delta_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden[self.hidden_input <= 0] = 0
        grad_weights_hidden_output = np.outer(self.hidden_layer, delta_output)
        grad_weights_input_hidden = np.outer(x, delta_hidden)
        self.update_weights(grad_weights_input_hidden, grad_weights_hidden_output)

    def update_weights(self, grad_wih, grad_who):
        self.weights_input_hidden -= self.learning_rate * grad_wih
        self.weights_hidden_output -= self.learning_rate * grad_who

class HiveMind:
    def __init__(self):
        self.weights_input_hidden_pool = []
        self.weights_hidden_output_pool = []
        self.lock = threading.Lock()
        self.global_knowledge = ""
        self.agents = []

    def collect_weights(self, weights_input_hidden, weights_hidden_output):
        with self.lock:
            self.weights_input_hidden_pool.append(weights_input_hidden)
            self.weights_hidden_output_pool.append(weights_hidden_output)

    def synchronize(self):
        with self.lock:
            if self.weights_input_hidden_pool and self.weights_hidden_output_pool:
                avg_inp_hid = np.mean(self.weights_input_hidden_pool, axis=0)
                avg_hid_out = np.mean(self.weights_hidden_output_pool, axis=0)
                self.weights_input_hidden_pool.clear()
                self.weights_hidden_output_pool.clear()
                return avg_inp_hid, avg_hid_out
            else:
                return None, None

    def update_global_knowledge(self, knowledge):
        with self.lock:
            self.global_knowledge += knowledge + '\n'

    def get_global_knowledge(self):
        with self.lock:
            return self.global_knowledge

    def save_models(self):
        with self.lock:
            for i, agent in enumerate(self.agents):
                with open(f'agent_{i}_model.pkl', 'wb') as f:
                    pickle.dump(agent.brain, f)

    def load_models(self):
        with self.lock:
            for i, agent in enumerate(self.agents):
                model_path = f'agent_{i}_model.pkl'
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        agent.brain = pickle.load(f)

class Agent:
    def __init__(self, agent_id, input_size, hidden_size, output_size, hive_mind, instruction):
        self.agent_id = agent_id
        self.brain = NeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            learning_rate=0.001
        )
        self.hive_mind = hive_mind
        self.running = True
        self.state = np.zeros(input_size)
        self.action_space = output_size
        self.driver = self.initialize_browser()
        self.instruction = instruction
        self.actions = {
            0: self.browse_website,
            1: self.click_button,
            2: self.fill_form,
            3: self.scroll_page,
            4: self.navigate_back,
            5: self.search_google
        }

    def initialize_browser(self):
        options = Options()
        driver = webdriver.Chrome(options=options)
        return driver

    def perceive(self):
        data = self.hive_mind.get_global_knowledge()
        if data:
            perception = self.process_data(data)
        else:
            perception = np.random.rand(self.brain.input_size)
        return perception

    def process_data(self, data):
        words = data.split()
        vector = np.zeros(self.brain.input_size)
        for i, word in enumerate(words):
            if i < self.brain.input_size:
                vector[i] = hash(word) % 1
        return vector

    def select_actions(self, state):
        action_probabilities = self.brain.forward(state)
        actions = np.argsort(action_probabilities)[-3:]
        return actions

    def perform_actions(self, actions):
        output_text = f"Agent {self.agent_id} performed actions: "
        for action in actions:
            if action in self.actions:
                self.actions[action]()
                output_text += f"{self.actions[action].__name__}, "
        logging.info(output_text.strip(', '))
        return output_text.strip(', ')

    def search_google(self):
        try:
            self.driver.get("https://www.google.com")
            search_box = self.driver.find_element(By.NAME, "q")
            search_query = self.instruction.split('search for', 1)[1].strip()
            search_box.send_keys(search_query)
            search_box.submit()
            time.sleep(2)
            self.extract_and_explore_links()
        except Exception as e:
            logging.error(f"Agent {self.agent_id} error in search_google: {e}")

    def extract_and_explore_links(self):
        try:
            results = self.driver.find_elements(By.CSS_SELECTOR, 'div.g')
            links = []
            for result in results[:5]:
                link = result.find_element(By.TAG_NAME, 'a').get_attribute('href')
                links.append(link)
            for link in links:
                self.driver.get(link)
                self.explore_page()
        except Exception as e:
            logging.error(f"Agent {self.agent_id} error in extract_and_explore_links: {e}")

    def browse_website(self):
        logging.info(f"Agent {self.agent_id} browsing...")

    def explore_page(self):
        try:
            page_text = self.driver.find_element(By.TAG_NAME, 'body').text
            self.hive_mind.update_global_knowledge(page_text)
            logging.info(f"Agent {self.agent_id} explored page and updated knowledge")
            self.scroll_page()
            self.click_button()
        except Exception as e:
            logging.error(f"Agent {self.agent_id} error in explore_page: {e}")

    def click_button(self):
        try:
            buttons = self.driver.find_elements(By.TAG_NAME, 'button')
            if buttons:
                buttons[0].click()
                logging.info(f"Agent {self.agent_id} clicked button")
        except Exception as e:
            logging.error(f"Agent {self.agent_id} error in click_button: {e}")

    def fill_form(self):
        try:
            inputs = self.driver.find_elements(By.TAG_NAME, 'input')
            for input_element in inputs:
                input_type = input_element.get_attribute('type')
                if input_type in ['text', 'email', 'password']:
                    input_element.send_keys('test')
                    logging.info(f"Agent {self.agent_id} filled form with 'test'")
        except Exception as e:
            logging.error(f"Agent {self.agent_id} error in fill_form: {e}")

    def scroll_page(self):
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            logging.info(f"Agent {self.agent_id} scrolled page")
        except Exception as e:
            logging.error(f"Agent {self.agent_id} error in scroll_page: {e}")

    def navigate_back(self):
        self.driver.back()

    def learn(self, state, actions, reward, next_state):
        target = self.brain.forward(state)
        for action in actions:
            target[action] = reward
        self.brain.train(state, target)
        self.hive_mind.collect_weights(
            self.brain.weights_input_hidden.copy(),
            self.brain.weights_hidden_output.copy()
        )

    def run(self):
        while self.running:
            state = self.perceive()
            actions = self.select_actions(state)
            output_text = self.perform_actions(actions)
            reward = self.evaluate_actions()
            next_state = self.perceive()
            self.learn(state, actions, reward, next_state)
            self.generate_output_text(output_text, reward)
            time.sleep(1)
            self.running = False  # Stop after one exploration sequence for testing

    def evaluate_actions(self):
        return 1

    def generate_output_text(self, actions_text, reward):
        learned_text = f"Agent {self.agent_id} learned from actions. Reward: {reward}"
        full_output = f"{actions_text}. {learned_text}"
        logging.info(full_output)

    def stop(self):
        self.running = False
        self.driver.quit()

def main():
    hive_mind = HiveMind()
    num_agents = 1
    input_size = 100
    hidden_size = 75
    output_size = 6
    test_instruction = "Go to google.com and search for best food in vestal, ny"
    agents = [
        Agent(
            agent_id=i,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            hive_mind=hive_mind,
            instruction=test_instruction
        )
        for i in range(num_agents)
    ]
    hive_mind.agents = agents
    threads = []
    for agent in agents:
        thread = threading.Thread(target=agent.run)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    try:
        for thread in threads:
            thread.join()
        logging.info(f"Final Global Knowledge:\n{hive_mind.get_global_knowledge()}")
    except KeyboardInterrupt:
        logging.info("Simulation interrupted by user.")
    finally:
        for agent in agents:
            agent.stop()
        logging.info("Simulation finished.")

if __name__ == "__main__":
    main()

