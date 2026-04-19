import asyncio


def empty_queue(queue: asyncio.Queue):
    while not queue.empty():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            break
