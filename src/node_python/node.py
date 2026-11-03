#!/usr/bin/env python3

import sys
import socket
import struct
import math
import time
import os
from concurrent.futures import ThreadPoolExecutor

SERVER_HOST = "localhost"
SERVER_PORT = 5001

CLASSES = ["gato", "mariposa", "perro"]

# Filtros convolucionales 3x3
KERNELS = [
    # 0: Sobel Horizontal
    [[-1.0, -2.0, -1.0], [ 0.0,  0.0,  0.0], [ 1.0,  2.0,  1.0]],
    # 1: Sobel Vertical
    [[-1.0,  0.0,  1.0], [-2.0,  0.0,  2.0], [-1.0,  0.0,  1.0]],
    # 2: Laplaciano / Ridge
    [[ 0.0,  1.0,  0.0], [ 1.0, -4.0,  1.0], [ 0.0,  1.0,  0.0]],
    # 3: Diagonal 45°
    [[-2.0, -1.0,  0.0], [-1.0,  0.0,  1.0], [ 0.0,  1.0,  2.0]],
    # 4: Sharpening
    [[ 0.0, -1.0,  0.0], [-1.0,  5.0, -1.0], [ 0.0, -1.0,  0.0]],
    # 5: Emboss
    [[-2.0, -1.0,  0.0], [-1.0,  1.0,  1.0], [ 0.0,  1.0,  2.0]]
]

class PythonCubeNode:
    def __init__(self, node_id="Nodo_Python_1"):
        self.node_id = node_id
        self.input_size = 566
        self.hidden1_size = 128
        self.hidden2_size = 64
        self.output_size = 3
        
        # Pesos en memoria
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.b3 = None

    def relu(self, x):
        return x if x > 0.0 else 0.0

    def relu_deriv(self, x):
        return 1.0 if x > 0.0 else 0.0

    def softmax(self, z):
        max_z = max(z)
        exp_z = [math.exp(v - max_z) for v in z]
        s = sum(exp_z)
        denom = s if s > 0.0 else 1.0
        return [v / denom for v in exp_z]

    def set_weights(self, in_size, h1, h2, out, W1, b1, W2, b2, W3, b3):
        self.input_size = in_size
        self.hidden1_size = h1
        self.hidden2_size = h2
        self.output_size = out
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3

    def _apply_single_kernel(self, k, gray):
        fmap = [[0.0] * 30 for _ in range(30)]
        for y in range(1, 31):
            for x in range(1, 31):
                s = (gray[y-1][x-1] * k[0][0] + gray[y-1][x] * k[0][1] + gray[y-1][x+1] * k[0][2] +
                     gray[y][x-1]   * k[1][0] + gray[y][x]   * k[1][1] + gray[y][x+1]   * k[1][2] +
                     gray[y+1][x-1] * k[2][0] + gray[y+1][x] * k[2][1] + gray[y+1][x+1] * k[2][2])
                fmap[y - 1][x - 1] = s if s > 0.0 else 0.0

        pooled = []
        for by in range(3):
            for bx in range(3):
                m = 0.0
                for py in range(by * 10, (by + 1) * 10):
                    for px in range(bx * 10, (bx + 1) * 10):
                        v = fmap[py][px]
                        if v > m:
                            m = v
                pooled.append(m)
        return pooled

    def compute_convolutions(self, flat_gray, color_hist):
        gray = [flat_gray[y * 32 : (y + 1) * 32] for y in range(32)]
        
        with ThreadPoolExecutor(max_workers=len(KERNELS)) as executor:
            kernel_pools = list(executor.map(lambda k: self._apply_single_kernel(k, gray), KERNELS))

        conv_feats = []
        conv_sum = 0.0
        for pool in kernel_pools:
            for val in pool:
                conv_feats.append(val)
                conv_sum += val

        if conv_sum > 0.0:
            conv_feats = [v / conv_sum for v in conv_feats]

        return list(color_hist) + conv_feats

    def predict_mlp(self, x):
        if not self.W1:
            raise ValueError("Pesos del modelo no inicializados en el nodo.")

        def eval_layer(weights, biases, inputs, h_size, act_fn):
            num_workers = 4
            chunk_size = (h_size + num_workers - 1) // num_workers
            chunks = [range(c * chunk_size, min((c + 1) * chunk_size, h_size)) for c in range(num_workers) if c * chunk_size < h_size]
            
            def chunk_calc(r):
                in_len = len(inputs)
                return [(i, act_fn(biases[i] + sum(weights[i][j] * inputs[j] for j in range(in_len)))) for i in r]

            out = [0.0] * h_size
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for chunk_res in executor.map(chunk_calc, chunks):
                    for idx, val in chunk_res:
                        out[idx] = val
            return out

        a1 = eval_layer(self.W1, self.b1, x, self.hidden1_size, self.relu)
        a2 = eval_layer(self.W2, self.b2, a1, self.hidden2_size, self.relu)

        z3 = [self.b3[i] + sum(self.W3[i][j] * a2[j] for j in range(self.hidden2_size))
              for i in range(self.output_size)]

        probs = self.softmax(z3)
        best_class = probs.index(max(probs))
        return probs, best_class

    def compute_batch_gradients(self, batch_data):
        dW1 = [[0.0] * self.input_size for _ in range(self.hidden1_size)]
        db1 = [0.0] * self.hidden1_size
        dW2 = [[0.0] * self.hidden1_size for _ in range(self.hidden2_size)]
        db2 = [0.0] * self.hidden2_size
        dW3 = [[0.0] * self.hidden2_size for _ in range(self.output_size)]
        db3 = [0.0] * self.output_size

        for x, label in batch_data:
            # Forward
            z1 = [self.b1[i] + sum(self.W1[i][j] * x[j] for j in range(self.input_size)) for i in range(self.hidden1_size)]
            a1 = [self.relu(v) for v in z1]

            z2 = [self.b2[i] + sum(self.W2[i][j] * a1[j] for j in range(self.hidden1_size)) for i in range(self.hidden2_size)]
            a2 = [self.relu(v) for v in z2]

            z3 = [self.b3[i] + sum(self.W3[i][j] * a2[j] for j in range(self.hidden2_size)) for i in range(self.output_size)]
            probs = self.softmax(z3)

            # Backprop output
            dz3 = [probs[i] - (1.0 if i == label else 0.0) for i in range(self.output_size)]

            # Backprop hidden 2
            dz2 = [0.0] * self.hidden2_size
            for j in range(self.hidden2_size):
                err = sum(dz3[i] * self.W3[i][j] for i in range(self.output_size))
                dz2[j] = err * self.relu_deriv(z2[j])

            # Backprop hidden 1
            dz1 = [0.0] * self.hidden1_size
            for j in range(self.hidden1_size):
                err = sum(dz2[i] * self.W2[i][j] for i in range(self.hidden2_size))
                dz1[j] = err * self.relu_deriv(z1[j])

            # Acumular gradientes
            for i in range(self.output_size):
                db3[i] += dz3[i]
                for j in range(self.hidden2_size):
                    dW3[i][j] += dz3[i] * a2[j]

            for i in range(self.hidden2_size):
                db2[i] += dz2[i]
                for j in range(self.hidden1_size):
                    dW2[i][j] += dz2[i] * a1[j]

            for i in range(self.hidden1_size):
                db1[i] += dz1[i]
                for j in range(self.input_size):
                    dW1[i][j] += dz1[i] * x[j]

        return len(batch_data), dW1, db1, dW2, db2, dW3, db3


# Funciones de red por Sockets

def read_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Socket cerrado antes de completar lectura")
        buf.extend(chunk)
    return buf

def read_utf(sock):
    raw_len = sock.recv(2)
    if not raw_len or len(raw_len) < 2:
        return None
    length = struct.unpack(">H", raw_len)[0]
    data = read_exact(sock, length)
    return data.decode("utf-8")

def write_utf(sock, s):
    encoded = s.encode("utf-8")
    sock.sendall(struct.pack(">H", len(encoded)) + encoded)

def read_int(sock):
    data = read_exact(sock, 4)
    return struct.unpack(">i", data)[0]

def write_int(sock, val):
    sock.sendall(struct.pack(">i", val))

def read_doubles(sock, count):
    raw = read_exact(sock, count * 8)
    return list(struct.unpack(f">{count}d", raw))

def write_doubles(sock, vals):
    count = len(vals)
    sock.sendall(struct.pack(f">{count}d", *vals))


def load_local_model_weights(filepath="modelo_animales.bin"):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        magic, in_size, h1, h2, out, lr = struct.unpack(">iiiii d", data[:28])
        if magic != 0x43554245: # "CUBE"
            return None
        offset = 28
        
        w1_count = h1 * in_size
        W1_flat = struct.unpack(f">{w1_count}d", data[offset : offset + w1_count * 8])
        offset += w1_count * 8
        W1 = [list(W1_flat[i * in_size : (i + 1) * in_size]) for i in range(h1)]
        
        b1 = list(struct.unpack(f">{h1}d", data[offset : offset + h1 * 8]))
        offset += h1 * 8

        w2_count = h2 * h1
        W2_flat = struct.unpack(f">{w2_count}d", data[offset : offset + w2_count * 8])
        offset += w2_count * 8
        W2 = [list(W2_flat[i * h1 : (i + 1) * h1]) for i in range(h2)]

        b2 = list(struct.unpack(f">{h2}d", data[offset : offset + h2 * 8]))
        offset += h2 * 8

        w3_count = out * h2
        W3_flat = struct.unpack(f">{w3_count}d", data[offset : offset + w3_count * 8])
        offset += w3_count * 8
        W3 = [list(W3_flat[i * h2 : (i + 1) * h2]) for i in range(out)]

        b3 = list(struct.unpack(f">{out}d", data[offset : offset + out * 8]))
        
        return in_size, h1, h2, out, W1, b1, W2, b2, W3, b3
    except Exception as e:
        print(f"[Nodo Python] Aviso cargando modelo local: {e}")
        return None


def run_node(host=SERVER_HOST, port=SERVER_PORT, node_id="Nodo_Python_1"):
    cube = PythonCubeNode(node_id)
    weights = load_local_model_weights()
    if weights:
        cube.set_weights(*weights)
        print(f"[{node_id}] Pesos del modelo 'modelo_animales.bin' cargados exitosamente.")

    print(f"[{node_id}] Conectando al Servidor Central en {host}:{port}...")

    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print(f"[{node_id}] ¡Conectado al Clúster!")

            # Registrar capacidades: Lenguaje PYTHON, micro-chunks CONV + MLP
            reg_msg = f"REGISTER:{node_id}:PYTHON:CONV,MLP"
            write_utf(sock, reg_msg)

            while True:
                cmd = read_utf(sock)
                if cmd is None or cmd == "QUIT":
                    print(f"[{node_id}] Servidor solicitó desconexión.")
                    break

                if cmd == "TASK_CLASSIFY":
                    gray_len = read_int(sock)
                    flat_gray = read_doubles(sock, gray_len)
                    hist_len = read_int(sock)
                    color_hist = read_doubles(sock, hist_len)

                    t0 = time.time()
                    features = cube.compute_convolutions(flat_gray, color_hist)
                    conv_ms = (time.time() - t0) * 1000

                    t1 = time.time()
                    probs, best_idx = cube.predict_mlp(features)
                    mlp_ms = (time.time() - t1) * 1000

                    best_name = CLASSES[best_idx].upper()
                    conf = probs[best_idx] * 100.0

                    write_utf(sock, f"RESULT_CLASSIFY:{node_id}:{best_name}:{conf:.2f}:{conv_ms:.2f}:{mlp_ms:.2f}")
                    write_int(sock, len(probs))
                    write_doubles(sock, probs)

                elif cmd == "TASK_CONV":
                    # Micro-chunk 1: Convolución
                    gray_len = read_int(sock) # 1024
                    flat_gray = read_doubles(sock, gray_len)
                    hist_len = read_int(sock) # 512
                    color_hist = read_doubles(sock, hist_len)

                    t0 = time.time()
                    features = cube.compute_convolutions(flat_gray, color_hist)
                    elapsed_ms = (time.time() - t0) * 1000

                    write_utf(sock, f"RESULT_CONV:{node_id}:{elapsed_ms:.2f}")
                    write_int(sock, len(features))
                    write_doubles(sock, features)
                    # print(f"[{node_id}] Micro-chunk CONV completado en {elapsed_ms:.2f} ms")

                elif cmd == "TASK_MLP":
                    # Micro-chunk 2: MLP Inferencia
                    feat_len = read_int(sock)
                    features = read_doubles(sock, feat_len)

                    t0 = time.time()
                    probs, best_idx = cube.predict_mlp(features)
                    elapsed_ms = (time.time() - t0) * 1000

                    best_name = CLASSES[best_idx].upper()
                    conf = probs[best_idx] * 100.0

                    write_utf(sock, f"RESULT_MLP:{node_id}:{best_name}:{conf:.2f}:{elapsed_ms:.2f}")
                    write_int(sock, len(probs))
                    write_doubles(sock, probs)
                    # print(f"[{node_id}] Micro-chunk MLP: {best_name} ({conf:.1f}%) en {elapsed_ms:.2f} ms")

                elif cmd == "TASK_TRAIN_BATCH":
                    # Entrenamiento distribuido: calcular gradientes
                    batch_size = read_int(sock)
                    batch = []
                    for _ in range(batch_size):
                        label = read_int(sock)
                        feat_len = read_int(sock)
                        x = read_doubles(sock, feat_len)
                        batch.append((x, label))

                    t0 = time.time()
                    count, dW1, db1, dW2, db2, dW3, db3 = cube.compute_batch_gradients(batch)
                    elapsed_ms = (time.time() - t0) * 1000

                    write_utf(sock, f"RESULT_TRAIN_BATCH:{node_id}:{count}:{elapsed_ms:.2f}")
                    # Enviar gradientes
                    write_int(sock, count)
                    
                    # Aplanar y enviar dW1, db1, dW2, db2, dW3, db3
                    flat_dW1 = [val for row in dW1 for val in row]
                    write_doubles(sock, flat_dW1)
                    write_doubles(sock, db1)

                    flat_dW2 = [val for row in dW2 for val in row]
                    write_doubles(sock, flat_dW2)
                    write_doubles(sock, db2)

                    flat_dW3 = [val for row in dW3 for val in row]
                    write_doubles(sock, flat_dW3)
                    write_doubles(sock, db3)

                elif cmd == "SYNC_WEIGHTS":
                    # Sincronización de pesos desde el Servidor Maestro
                    in_size = read_int(sock)
                    h1 = read_int(sock)
                    h2 = read_int(sock)
                    out = read_int(sock)

                    w1_flat = read_doubles(sock, h1 * in_size)
                    W1 = [w1_flat[i * in_size : (i + 1) * in_size] for i in range(h1)]
                    b1 = read_doubles(sock, h1)

                    w2_flat = read_doubles(sock, h2 * h1)
                    W2 = [w2_flat[i * h1 : (i + 1) * h1] for i in range(h2)]
                    b2 = read_doubles(sock, h2)

                    w3_flat = read_doubles(sock, out * h2)
                    W3 = [w3_flat[i * h2 : (i + 1) * h2] for i in range(out)]
                    b3 = read_doubles(sock, out)

                    cube.set_weights(in_size, h1, h2, out, W1, b1, W2, b2, W3, b3)
                    write_utf(sock, "SYNC_OK")
                    print(f"[{node_id}] Pesos del clúster sincronizados exitosamente.")

                elif cmd == "PING":
                    write_utf(sock, "PONG")

            sock.close()
        except ConnectionRefusedError:
            print(f"[{node_id}] No se pudo conectar al servidor. Reintentando en 3s...")
            time.sleep(3)
        except Exception as e:
            print(f"[{node_id}] Error en conexión: {e}. Reconectando en 3s...")
            time.sleep(3)

if __name__ == "__main__":
    node_name = sys.argv[1] if len(sys.argv) > 1 else "Nodo_Python_1"
    host = sys.argv[2] if len(sys.argv) > 2 else SERVER_HOST
    port = int(sys.argv[3]) if len(sys.argv) > 3 else SERVER_PORT
    run_node(host, port, node_name)
