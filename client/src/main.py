import asyncio
import socket

import numpy as np
import sounddevice as sd

HOST = "127.0.0.1"
PORT = 9000


async def inputstream_generator(channels=1, **kwargs):
    q_in = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(callback=callback, channels=channels, **kwargs)

    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


# ---------------- SEND AUDIO ----------------
async def send_audio(sock: socket.socket):
    loop = asyncio.get_running_loop()
    async for indata, _ in inputstream_generator(samplerate=16000, dtype="int16"):
        await loop.sock_sendall(sock, indata.tobytes())


# ---------------- RECEIVE + PLAY ----------------
async def receive_and_play(sock):
    loop = asyncio.get_running_loop()

    # Match the server audio: float32, mono
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype="float32")
    stream.start()

    try:
        while True:
            data = await loop.sock_recv(sock, 4096)
            if not data:
                break

            # Interpret the bytes as float32, not int16
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            stream.write(audio_chunk)

    except ConnectionResetError:
        # Server closed abruptly → normal end of audio
        pass

    finally:
        stream.stop()
        stream.close()


# ---------------- MAIN ----------------
async def main():
    loop = asyncio.get_running_loop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    await loop.sock_connect(sock, (HOST, PORT))

    send_task = asyncio.create_task(send_audio(sock))
    recv_task = asyncio.create_task(receive_and_play(sock))

    try:
        await asyncio.gather(send_task, recv_task)
    finally:
        send_task.cancel()
        recv_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
