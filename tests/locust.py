from locust import HttpUser, task, between
import numpy as np
import random

# здесь находятся 1000 сообщений из валидационного датасета,
# которые будут использоваться для нагрузочного тестирования
MESSAGES = np.load("messages.npy", allow_pickle=True).tolist()


class MessageUser(HttpUser):

    wait_time = between(0.1, 0.7)

    @task
    def predict(self):
        text = random.choice(MESSAGES)
        self.client.post(
            "/predict",
            json={
                "text": text
            },
            name="/predict"
        )