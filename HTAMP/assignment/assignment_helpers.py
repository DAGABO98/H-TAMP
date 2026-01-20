import heapq

class TaskQueue:
    def __init__(self):
        self.heap = []

    def add_task(self, priority, task_id):
        heapq.heappush(self.heap, (priority, task_id))

    def pop_task(self):
        return heapq.heappop(self.heap)[1] if self.heap else None

class AssignmentHelper:

    @staticmethod
    def insertion_procedure():
        pass
    