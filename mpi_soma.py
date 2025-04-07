import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_in_chunks(file_path, lines_per_chunk=10):
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= lines_per_chunk:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def main():
    if rank == 0:
        total_sum = 0
        worker_index = 1
        workers = list(range(1, size))
        print("Process 0 started reading the file.")
        for chunk in read_in_chunks('input.txt', lines_per_chunk=2):
            data = [int(line.strip()) for line in chunk]
            dest = workers[worker_index % len(workers)]
            comm.send(data, dest=dest)
            print(f"Process 0 sent data to process {dest}: {data}")

            result = comm.recv(source=dest)
            print(f"Process 0 received result from process {dest}: {result}")
            total_sum += result
            worker_index += 1
        for i in workers:
            comm.send(None, dest=i)

        print(f"Final total sum: {total_sum}")

    else:
        print(f"Process {rank} waiting for data.")
        while True:
            data = comm.recv(source=0)
            if data is None:
                print(f"Process {rank} received termination signal.")
                break
            print(f"Process {rank} received data: {data}")
            result = sum(data)
            print(f"Process {rank} computed sum: {result}")
            comm.send(result, dest=0)

if __name__ == "__main__":
    main()
