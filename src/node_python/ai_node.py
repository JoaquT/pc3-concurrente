#!/usr/bin/env python3
"""
Open AI Cubes - Nodo IA (Python / Alpine)
Arquitectura de Red Neuronal Convolucional (CNN) desde CERO (Sin Frameworks de IA)
Comunicación exclusiva mediante Middleware RabbitMQ (pika).
"""

import json
import math
import os
import random
import struct
import time
import pika

# =====================================================================
# 1. CAPAS DE LA RED NEURONAL CONVOLUCIONAL (DESDE CERO)
# =====================================================================

class Conv2DLayer:
    """
    Capa de Convolución 2D implementada desde cero con listas nativas.
    Aplica un banco de filtros (kernels) de tamaño KxK sobre una imagen 2D.
    """
    def __init__(self, kernels=None):
        if kernels is not None:
            self.kernels = kernels
        else:
            # 6 filtros clásicos de extracción de características 3x3 (Sobel, Laplaciano, etc.)
            self.kernels = [
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

    def forward(self, image_2d):
        """
        Ejecuta el paso hacia adelante (Forward Pass) de la convolución.
        :param image_2d: Matriz 2D [H][W] con intensidades de píxeles normalizadas.
        :return: Lista de mapas de características 3D [num_kernels][H - K + 1][W - K + 1].
        """
        h = len(image_2d)
        w = len(image_2d[0])
        k_size = len(self.kernels[0])
        out_h = h - k_size + 1
        out_w = w - k_size + 1

        feature_maps = []
        for kernel in self.kernels:
            fmap = [[0.0] * out_w for _ in range(out_h)]
            for y in range(out_h):
                for x in range(out_w):
                    acc = 0.0
                    for ky in range(k_size):
                        for kx in range(k_size):
                            acc += image_2d[y + ky][x + kx] * kernel[ky][kx]
                    fmap[y][x] = acc
            feature_maps.append(fmap)
        return feature_maps


class ReLULayer:
    """
    Capa de Activación No Lineal ReLU: f(x) = max(0, x).
    Soporta tanto mapas de características 3D como vectores 1D.
    """
    def forward_3d(self, feature_maps):
        out_maps = []
        for fmap in feature_maps:
            out_fmap = [[val if val > 0.0 else 0.0 for val in row] for row in fmap]
            out_maps.append(out_fmap)
        return out_maps

    def forward_1d(self, vector):
        return [val if val > 0.0 else 0.0 for val in vector]


class MaxPooling2DLayer:
    """
    Capa de Submuestreo Espacial (Max Pooling 2D) desde cero.
    Reduce la dimensionalidad reteniendo el valor máximo en cada ventana (pool_size x stride).
    """
    def __init__(self, pool_size=10, stride=10):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, feature_maps):
        pooled_maps = []
        for fmap in feature_maps:
            h = len(fmap)
            w = len(fmap[0])
            out_h = h // self.stride
            out_w = w // self.stride
            pooled = [[0.0] * out_w for _ in range(out_h)]

            for by in range(out_h):
                for bx in range(out_w):
                    max_val = 0.0
                    start_y = by * self.stride
                    start_x = bx * self.stride
                    for py in range(start_y, min(start_y + self.pool_size, h)):
                        for px in range(start_x, min(start_x + self.pool_size, w)):
                            val = fmap[py][px]
                            if val > max_val:
                                max_val = val
                    pooled[by][bx] = max_val
            pooled_maps.append(pooled)
        return pooled_maps


class FlattenLayer:
    """
    Capa Flatten: Aplana los mapas de características 3D a un vector 1D normalizado
    y permite concatenar descriptores adicionales (como histograma de color).
    """
    def forward(self, pooled_maps, extra_features=None):
        flat_conv = []
        conv_sum = 0.0
        for pmap in pooled_maps:
            for row in pmap:
                for val in row:
                    flat_conv.append(val)
                    conv_sum += val

        if conv_sum > 0.0:
            flat_conv = [v / conv_sum for v in flat_conv]

        if extra_features:
            return list(extra_features) + flat_conv
        return flat_conv


class DenseLayer:
    """
    Capa Densa (Fully Connected): Z = W * X + b
    Implementada desde cero con arreglos nativos.
    """
    def __init__(self, input_size, output_size, weights=None, biases=None, seed=42):
        self.input_size = input_size
        self.output_size = output_size

        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            # Inicialización He / Xavier determinista sin librerías externas
            rng = random.Random(seed)
            limit = math.sqrt(2.0 / max(1, input_size))
            self.weights = [
                [rng.uniform(-limit, limit) for _ in range(input_size)]
                for _ in range(output_size)
            ]
            self.biases = [0.0 for _ in range(output_size)]

    def forward(self, x):
        out = [0.0] * self.output_size
        for i in range(self.output_size):
            w_row = self.weights[i]
            s = self.biases[i]
            for j in range(self.input_size):
                s += w_row[j] * x[j]
            out[i] = s
        return out


class SoftmaxLayer:
    """
    Capa de Activación Softmax numéricamente estable para distribución de probabilidades.
    """
    def forward(self, logits):
        max_z = max(logits)
        exp_z = [math.exp(z - max_z) for z in logits]
        sum_exp = sum(exp_z)
        denom = sum_exp if sum_exp > 0.0 else 1.0
        return [val / denom for val in exp_z]


# =====================================================================
# 2. ARQUITECTURA COMPLETA DE LA CNN (FORWARD PASS)
# =====================================================================

class ConvolutionalNeuralNetwork:
    """
    Red Neuronal Convolucional completa desde cero:
    Conv2D -> ReLU -> MaxPooling2D -> Flatten -> Dense1 -> ReLU -> Dense2 -> ReLU -> Dense3 -> Softmax
    """
    CLASSES = ["GATO", "MARIPOSA", "PERRO"]

    def __init__(self, model_path="modelo_animales.bin"):
        self.conv = Conv2DLayer()
        self.relu = ReLULayer()
        # Imagen 32x32 -> Conv 3x3 válida -> 30x30 -> MaxPool 10x10 -> 3x3 (6 filtros * 9 = 54 features)
        self.pool = MaxPooling2DLayer(pool_size=10, stride=10)
        self.flatten = FlattenLayer()

        # Dimensiones por defecto: 512 (histograma RGB 8x8x8) + 54 (Conv features) = 566
        self.input_size = 566
        self.hidden1_size = 128
        self.hidden2_size = 64
        self.output_size = 3

        self.dense1 = DenseLayer(self.input_size, self.hidden1_size, seed=101)
        self.dense2 = DenseLayer(self.hidden1_size, self.hidden2_size, seed=202)
        self.dense3 = DenseLayer(self.hidden2_size, self.output_size, seed=303)
        self.softmax = SoftmaxLayer()

        self._try_load_weights(model_path)

    def _try_load_weights(self, filepath):
        """Carga pesos pre-entrenados desde archivo binario nativo si existe."""
        if not filepath or not os.path.exists(filepath):
            print(f"[CNN] Archivo '{filepath}' no encontrado. Usando pesos inicializados por defecto.")
            return
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            magic, in_size, h1, h2, out, _ = struct.unpack(">iiiiid", data[:28])
            if magic != 0x43554245:  # Magic header "CUBE"
                return
            offset = 28

            w1_count = h1 * in_size
            w1_flat = struct.unpack(f">{w1_count}d", data[offset:offset + w1_count * 8])
            offset += w1_count * 8
            W1 = [list(w1_flat[i * in_size:(i + 1) * in_size]) for i in range(h1)]
            b1 = list(struct.unpack(f">{h1}d", data[offset:offset + h1 * 8]))
            offset += h1 * 8

            w2_count = h2 * h1
            w2_flat = struct.unpack(f">{w2_count}d", data[offset:offset + w2_count * 8])
            offset += w2_count * 8
            W2 = [list(w2_flat[i * h1:(i + 1) * h1]) for i in range(h2)]
            b2 = list(struct.unpack(f">{h2}d", data[offset:offset + h2 * 8]))
            offset += h2 * 8

            w3_count = out * h2
            w3_flat = struct.unpack(f">{w3_count}d", data[offset:offset + w3_count * 8])
            offset += w3_count * 8
            W3 = [list(w3_flat[i * h2:(i + 1) * h2]) for i in range(out)]
            b3 = list(struct.unpack(f">{out}d", data[offset:offset + out * 8]))

            self.input_size = in_size
            self.hidden1_size = h1
            self.hidden2_size = h2
            self.output_size = out

            self.dense1 = DenseLayer(in_size, h1, W1, b1)
            self.dense2 = DenseLayer(h1, h2, W2, b2)
            self.dense3 = DenseLayer(h2, out, W3, b3)
            print(f"[CNN] Pesos cargados exitosamente desde '{filepath}' ({in_size}-> {h1}-> {h2}-> {out}).")
        except Exception as e:
            print(f"[CNN] Aviso al leer '{filepath}': {e}. Usando pesos inicializados.")

    def forward(self, flat_gray_32x32, color_hist_512=None):
        """
        Ejecuta el Forward Pass completo de la CNN sobre una imagen 32x32.
        :param flat_gray_32x32: Lista de 1024 valores (32x32) en escala de grises [0..1].
        :param color_hist_512: Lista opcional de 512 valores del histograma de color.
        :return: Diccionario con la clase predicha, confianza y probabilidades por clase.
        """
        t0 = time.time()

        # Asegurar tamaño 32x32 (1024 píxeles)
        if len(flat_gray_32x32) < 1024:
            flat_gray_32x32 = list(flat_gray_32x32) + [0.0] * (1024 - len(flat_gray_32x32))
        image_2d = [flat_gray_32x32[y * 32:(y + 1) * 32] for y in range(32)]

        if color_hist_512 is None:
            color_hist_512 = [0.0] * 512
        elif len(color_hist_512) < 512:
            color_hist_512 = list(color_hist_512) + [0.0] * (512 - len(color_hist_512))

        # Paso 1: Convolución 2D
        conv_maps = self.conv.forward(image_2d)

        # Paso 2: Activación ReLU sobre mapas convolucionales
        relu_maps = self.relu.forward_3d(conv_maps)

        # Paso 3: Max Pooling 2D
        pooled_maps = self.pool.forward(relu_maps)

        # Paso 4: Flatten (+ concatenación de descriptor cromático si aplica)
        extra = color_hist_512 if self.input_size == 566 else None
        flat_vector = self.flatten.forward(pooled_maps, extra_features=extra)

        # Ajustar vector si el modelo cargado tiene un input_size distinto
        if len(flat_vector) < self.input_size:
            flat_vector = flat_vector + [0.0] * (self.input_size - len(flat_vector))
        elif len(flat_vector) > self.input_size:
            flat_vector = flat_vector[:self.input_size]

        # Paso 5: Capas Densas (MLP Clasificador) + ReLU
        z1 = self.dense1.forward(flat_vector)
        a1 = self.relu.forward_1d(z1)

        z2 = self.dense2.forward(a1)
        a2 = self.relu.forward_1d(z2)

        logits = self.dense3.forward(a2)

        # Paso 6: Softmax
        probs = self.softmax.forward(logits)
        best_idx = probs.index(max(probs))
        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "predicted_class": self.CLASSES[best_idx],
            "confidence": round(probs[best_idx] * 100.0, 2),
            "probabilities": {
                self.CLASSES[i]: round(probs[i], 4) for i in range(len(self.CLASSES))
            },
            "inference_time_ms": round(elapsed_ms, 2)
        }


# =====================================================================
# 3. NODO TRABAJADOR DE IA CON MIDDLEWARE RABBITMQ (PIKA)
# =====================================================================

class RabbitMQAINode:
    """
    Nodo consumidor de RabbitMQ que recibe tareas de clasificación de imágenes,
    ejecuta el Forward Pass de la CNN desde cero y publica la respuesta.
    """
    QUEUE_IN = "ai_tasks_queue"
    QUEUE_OUT = "ai_results_queue"

    def __init__(self, node_id="Nodo_IA_Python_1", rabbit_host="localhost", rabbit_port=5672,
                 rabbit_user="guest", rabbit_pass="guest", model_path="modelo_animales.bin"):
        self.node_id = node_id
        self.rabbit_host = rabbit_host
        self.rabbit_port = rabbit_port
        self.rabbit_user = rabbit_user
        self.rabbit_pass = rabbit_pass
        self.cnn = ConvolutionalNeuralNetwork(model_path=model_path)

    def _on_message(self, ch, method, properties, body):
        """Callback ejecutado al recibir un mensaje en 'ai_tasks_queue'."""
        try:
            payload = json.loads(body.decode("utf-8"))
            request_id = payload.get("request_id", "unknown")
            flat_gray = payload.get("gray_pixels", [])
            color_hist = payload.get("color_histogram", None)

            print(f"[{self.node_id}] Procesando tarea de clasificación (ID: {request_id})...")
            prediction = self.cnn.forward(flat_gray, color_hist)

            response = {
                "status": "OK",
                "node_id": self.node_id,
                "request_id": request_id,
                "result": prediction
            }
        except Exception as e:
            response = {
                "status": "ERROR",
                "node_id": self.node_id,
                "error": str(e)
            }

        response_bytes = json.dumps(response).encode("utf-8")

        # Si el productor especificó cola de retorno (RPC reply_to), respondemos ahí;
        # de lo contrario, publicamos en la cola de resultados general 'ai_results_queue'.
        target_queue = properties.reply_to if (properties and properties.reply_to) else self.QUEUE_OUT
        corr_id = properties.correlation_id if properties else None

        ch.basic_publish(
            exchange="",
            routing_key=target_queue,
            properties=pika.BasicProperties(
                correlation_id=corr_id,
                content_type="application/json",
                delivery_mode=2
            ),
            body=response_bytes
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[{self.node_id}] Resultado enviado a '{target_queue}': {response.get('result', response)}")

    def start(self):
        credentials = pika.PlainCredentials(self.rabbit_user, self.rabbit_pass)
        params = pika.ConnectionParameters(
            host=self.rabbit_host,
            port=self.rabbit_port,
            credentials=credentials,
            heartbeat=60,
            blocked_connection_timeout=300
        )

        while True:
            try:
                print(f"[{self.node_id}] Conectando a RabbitMQ en {self.rabbit_host}:{self.rabbit_port}...")
                connection = pika.BlockingConnection(params)
                channel = connection.channel()

                channel.queue_declare(queue=self.QUEUE_IN, durable=True)
                channel.queue_declare(queue=self.QUEUE_OUT, durable=True)
                channel.basic_qos(prefetch_count=1)

                channel.basic_consume(
                    queue=self.QUEUE_IN,
                    on_message_callback=self._on_message,
                    auto_ack=False
                )

                print(f"[{self.node_id}] Esperando mensajes en la cola '{self.QUEUE_IN}'...")
                channel.start_consuming()
            except pika.exceptions.AMQPConnectionError as e:
                print(f"[{self.node_id}] RabbitMQ no disponible ({e}). Reintentando en 4 segundos...")
                time.sleep(4)
            except KeyboardInterrupt:
                print(f"\n[{self.node_id}] Nodo detenido manualmente.")
                break


if __name__ == "__main__":
    rabbit_host = os.getenv("RABBITMQ_HOST", "localhost")
    rabbit_port = int(os.getenv("RABBITMQ_PORT", "5672"))
    rabbit_user = os.getenv("RABBITMQ_USER", "guest")
    rabbit_pass = os.getenv("RABBITMQ_PASS", "guest")
    node_id = os.getenv("NODE_ID", "Nodo_IA_Python_1")
    model_file = os.getenv("MODEL_PATH", "modelo_animales.bin")

    node = RabbitMQAINode(
        node_id=node_id,
        rabbit_host=rabbit_host,
        rabbit_port=rabbit_port,
        rabbit_user=rabbit_user,
        rabbit_pass=rabbit_pass,
        model_path=model_file
    )
    node.start()
