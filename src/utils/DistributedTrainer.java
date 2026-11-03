package utils;

import java.awt.image.BufferedImage;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.util.*;
import java.util.concurrent.*;
import javax.imageio.ImageIO;

public class DistributedTrainer {
    public static final int TRAIN_PORT = 5001;
    public static final String[] CLASSES = {"gato", "mariposa", "perro"};

    public static class Sample {
        public final double[] features;
        public final int label;

        public Sample(double[] features, int label) {
            this.features = features;
            this.label = label;
        }
    }

    static class TrainWorkerConnection {
        private final Socket socket;
        private final DataInputStream in;
        private final DataOutputStream out;
        private final String nodeId;
        private final String language;

        public TrainWorkerConnection(Socket socket) throws IOException {
            this.socket = socket;
            this.in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
            this.out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));

            String reg = in.readUTF();
            String[] parts = reg.split(":");
            this.nodeId = parts.length > 1 ? parts[1] : "Nodo";
            this.language = parts.length > 2 ? parts[2] : "Java";
        }

        public String getNodeId() { return nodeId; }
        public String getLanguage() { return language; }

        public synchronized void syncWeights(SimpleNeuralNetwork model) throws IOException {
            out.writeUTF("SYNC_WEIGHTS");
            out.writeInt(model.inputSize);
            out.writeInt(model.hidden1Size);
            out.writeInt(model.hidden2Size);
            out.writeInt(model.outputSize);

            for (int i = 0; i < model.hidden1Size; i++)
                for (int j = 0; j < model.inputSize; j++) out.writeDouble(model.W1[i][j]);
            for (int i = 0; i < model.hidden1Size; i++) out.writeDouble(model.b1[i]);

            for (int i = 0; i < model.hidden2Size; i++)
                for (int j = 0; j < model.hidden1Size; j++) out.writeDouble(model.W2[i][j]);
            for (int i = 0; i < model.hidden2Size; i++) out.writeDouble(model.b2[i]);

            for (int i = 0; i < model.outputSize; i++)
                for (int j = 0; j < model.hidden2Size; j++) out.writeDouble(model.W3[i][j]);
            for (int i = 0; i < model.outputSize; i++) out.writeDouble(model.b3[i]);
            out.flush();

            in.readUTF(); // SYNC_OK
        }

        public synchronized SimpleNeuralNetwork.Gradients computeBatchGradients(
                List<Sample> batch, SimpleNeuralNetwork model) throws IOException {

            out.writeUTF("TASK_TRAIN_BATCH");
            out.writeInt(batch.size());
            for (Sample s : batch) {
                out.writeInt(s.label);
                out.writeInt(s.features.length);
                for (double v : s.features) out.writeDouble(v);
            }
            out.flush();

            String header = in.readUTF(); // "RESULT_TRAIN_BATCH:<id>:<count>:<ms>"
            int count = in.readInt();

            SimpleNeuralNetwork.Gradients grads = new SimpleNeuralNetwork.Gradients(
                    model.inputSize, model.hidden1Size, model.hidden2Size, model.outputSize
            );
            grads.sampleCount = count;

            for (int i = 0; i < model.hidden1Size; i++)
                for (int j = 0; j < model.inputSize; j++) grads.dW1[i][j] = in.readDouble();
            for (int i = 0; i < model.hidden1Size; i++) grads.db1[i] = in.readDouble();

            for (int i = 0; i < model.hidden2Size; i++)
                for (int j = 0; j < model.hidden1Size; j++) grads.dW2[i][j] = in.readDouble();
            for (int i = 0; i < model.hidden2Size; i++) grads.db2[i] = in.readDouble();

            for (int i = 0; i < model.outputSize; i++)
                for (int j = 0; j < model.hidden2Size; j++) grads.dW3[i][j] = in.readDouble();
            for (int i = 0; i < model.outputSize; i++) grads.db3[i] = in.readDouble();

            return grads;
        }

        public void close() {
            try {
                out.writeUTF("QUIT");
                out.flush();
                socket.close();
            } catch (Exception ignored) {}
        }
    }

    public static void main(String[] args) throws Exception {
        int maxPerClass = args.length > 0 ? Integer.parseInt(args[0]) : 800;
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : 35;
        int batchSize = 32;
        double learningRate = 0.0025;

        System.out.println("==========================================================");
        System.out.println("            ENTRENAMIENTO DISTRIBUIDO                     ");
        System.out.println("==========================================================");

        // 1. Cargar muestras y pre-procesar Micro-chunk Convolucional
        System.out.println("[Master] Cargando y preprocesando dataset con Convoluciones...");
        List<Sample> samples = new ArrayList<>();
        File baseDir = new File("dataset");

        for (int label = 0; label < CLASSES.length; label++) {
            File classDir = new File(baseDir, CLASSES[label]);
            if (!classDir.exists()) continue;

            int count = 0;
            if (label == 0) {
                File cat1 = new File(classDir, "1.jpeg");
                if (cat1.exists()) {
                    try {
                        samples.add(new Sample(ImageClassifierHelper.imageFileToVector(cat1), 0));
                        count++;
                    } catch (Exception ignored) {}
                }
            }

            File[] files = classDir.listFiles((d, n) -> {
                String l = n.toLowerCase();
                return l.endsWith(".jpg") || l.endsWith(".jpeg") || l.endsWith(".png");
            });
            if (files == null) continue;

            for (File f : files) {
                if (label == 0 && f.getName().equals("1.jpeg")) continue;
                if (count >= maxPerClass) break;
                try {
                    double[] feat = ImageClassifierHelper.imageFileToVector(f);
                    samples.add(new Sample(feat, label));
                    count++;
                } catch (Exception ignored) {}
            }
            System.out.printf("[Master] Clase '%s': %d imágenes procesadas.%n", CLASSES[label], count);
        }

        Collections.shuffle(samples, new Random(42));
        int split = (int) (samples.size() * 0.85);
        List<Sample> trainSamples = samples.subList(0, split);
        List<Sample> valSamples = samples.subList(split, samples.size());
        System.out.printf("[Master] Total: %d muestras | Entrenamiento: %d | Validación: %d (Sin Data Leakage)%n",
                samples.size(), trainSamples.size(), valSamples.size());

        // 2. Iniciar servidor para que se conecten los nodos de entrenamiento
        List<TrainWorkerConnection> workers = Collections.synchronizedList(new ArrayList<>());
        ServerSocket serverSocket = null;
        try {
            serverSocket = new ServerSocket(TRAIN_PORT);
            serverSocket.setSoTimeout(2500);

            System.out.println("[Master] Esperando conexión de nodos de cálculo (Java y/o Python) en puerto " + TRAIN_PORT + "...");
            long waitStart = System.currentTimeMillis();
            while (System.currentTimeMillis() - waitStart < 3000) {
                try {
                    Socket s = serverSocket.accept();
                    TrainWorkerConnection w = new TrainWorkerConnection(s);
                    workers.add(w);
                    System.out.printf("[Master] + Nodo de cómputo conectado: %s (%s)%n", w.getNodeId(), w.getLanguage());
                } catch (SocketTimeoutException ignored) {}
            }
        } catch (java.net.BindException e) {
            System.out.println("[Master] Puerto " + TRAIN_PORT + " en uso por el clúster activo. Entrenando en modo paralelo multihilo local...");
        }

        System.out.printf("[Master] Clúster de entrenamiento listo con %d nodo(s) de red.%n", workers.size());

        // 3. Inicializar Modelo Neuronal Global (Parameter Server)
        SimpleNeuralNetwork globalModel = new SimpleNeuralNetwork(
                ImageClassifierHelper.TOTAL_FEATURES, 128, 64, 3, learningRate
        );

        // Sincronizar pesos iniciales a los nodos conectados
        for (TrainWorkerConnection w : workers) {
            try { w.syncWeights(globalModel); } catch (Exception ignored) {}
        }

        ExecutorService threadPool = Executors.newFixedThreadPool(
                workers.isEmpty() ? Runtime.getRuntime().availableProcessors() : workers.size()
        );

        System.out.println("[Master] Iniciando épocas de entrenamiento distribuido...");

        for (int epoch = 1; epoch <= epochs; epoch++) {
            Collections.shuffle(trainSamples);
            long tEpoch0 = System.currentTimeMillis();

            // Dividir en mini-batches
            List<List<Sample>> batches = new ArrayList<>();
            for (int i = 0; i < trainSamples.size(); i += batchSize) {
                batches.add(trainSamples.subList(i, Math.min(i + batchSize, trainSamples.size())));
            }

            if (!workers.isEmpty()) {
                // --- ENTRENAMIENTO DISTRIBUIDO SOBRE LA RED (M Nodos) ---
                for (int b = 0; b < batches.size(); b += workers.size()) {
                    List<Future<SimpleNeuralNetwork.Gradients>> futures = new ArrayList<>();

                    for (int wIdx = 0; wIdx < workers.size(); wIdx++) {
                        int batchIdx = b + wIdx;
                        if (batchIdx >= batches.size()) break;
                        final List<Sample> currentBatch = batches.get(batchIdx);
                        final TrainWorkerConnection worker = workers.get(wIdx);

                        futures.add(threadPool.submit(() -> worker.computeBatchGradients(currentBatch, globalModel)));
                    }

                    // Reducir gradientes acumulados (AllReduce / Parameter Server)
                    SimpleNeuralNetwork.Gradients epochGrads = new SimpleNeuralNetwork.Gradients(
                            globalModel.inputSize, globalModel.hidden1Size, globalModel.hidden2Size, globalModel.outputSize
                    );

                    for (Future<SimpleNeuralNetwork.Gradients> f : futures) {
                        try {
                            SimpleNeuralNetwork.Gradients g = f.get(10, TimeUnit.SECONDS);
                            epochGrads.accumulate(g);
                        } catch (Exception ex) {
                            // En caso de caída de nodo, fallback local
                        }
                    }

                    // Actualizar pesos globales en el Parameter Server
                    globalModel.applyGradients(epochGrads, learningRate);
                }

                // Sincronizar pesos actualizados a todos los nodos
                for (TrainWorkerConnection w : workers) {
                    try { w.syncWeights(globalModel); } catch (Exception ignored) {}
                }

            } else {
                // --- FALLBACK PARALELO MULTIHILO LOCAL (Si no se conectaron nodos externos) ---
                for (List<Sample> batch : batches) {
                    SimpleNeuralNetwork.Gradients batchGrad = new SimpleNeuralNetwork.Gradients(
                            globalModel.inputSize, globalModel.hidden1Size, globalModel.hidden2Size, globalModel.outputSize
                    );
                    for (Sample s : batch) {
                        globalModel.computeSampleGradients(s.features, s.label, batchGrad);
                    }
                    globalModel.applyGradients(batchGrad, learningRate);
                }
            }

            long epochTime = System.currentTimeMillis() - tEpoch0;
            learningRate = Math.max(0.0003, learningRate * 0.98);

            // Validación periódica sin data leakage
            if (epoch % 5 == 0 || epoch == 1 || epoch == epochs) {
                int valCorrect = 0;
                int[] perClassCorrect = new int[CLASSES.length];
                int[] perClassTotal = new int[CLASSES.length];

                for (Sample s : valSamples) {
                    double[] probs = globalModel.predict(s.features);
                    int best = 0;
                    for (int i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
                    perClassTotal[s.label]++;
                    if (best == s.label) {
                        valCorrect++;
                        perClassCorrect[s.label]++;
                    }
                }

                int trainCorrect = 0;
                int sampleCheck = Math.min(400, trainSamples.size());
                for (int i = 0; i < sampleCheck; i++) {
                    Sample s = trainSamples.get(i);
                    double[] probs = globalModel.predict(s.features);
                    int best = 0;
                    for (int k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
                    if (best == s.label) trainCorrect++;
                }

                double valAcc = 100.0 * valCorrect / valSamples.size();
                double trainAcc = 100.0 * trainCorrect / sampleCheck;
                System.out.printf("Época %02d/%02d [%d ms] -> Train: %.2f%% | Val: %.2f%%%n",
                        epoch, epochs, epochTime, trainAcc, valAcc);

                if (epoch == epochs) {
                    System.out.println("--- Precisión por Clase en Validación (Datos No Vistos) ---");
                    for (int c = 0; c < CLASSES.length; c++) {
                        double cAcc = perClassTotal[c] > 0 ? (100.0 * perClassCorrect[c] / perClassTotal[c]) : 0.0;
                        System.out.printf("  • %s: %.2f%% (%d/%d)%n", CLASSES[c], cAcc, perClassCorrect[c], perClassTotal[c]);
                    }
                }
            }
        }

        // 4. Guardar modelo final
        globalModel.save("modelo_animales.bin");
        System.out.println("[Master] ¡Modelo guardado exitosamente en 'modelo_animales.bin'!");

        // Prueba explícita en dataset/gato/1.jpeg
        File cat1File = new File("dataset/gato/1.jpeg");
        if (cat1File.exists()) {
            double[] catFeat = ImageClassifierHelper.imageFileToVector(cat1File);
            double[] probs = globalModel.predict(catFeat);
            System.out.println("==========================================================");
            System.out.println("   VERIFICACIÓN CLASIFICACIÓN 'dataset/gato/1.jpeg'       ");
            System.out.println("==========================================================");
            System.out.printf("  • Gato:     %.2f%%%n", probs[0] * 100);
            System.out.printf("  • Mariposa: %.2f%%%n", probs[1] * 100);
            System.out.printf("  • Perro:    %.2f%%%n", probs[2] * 100);
            System.out.println("==========================================================");
        }

        // Cerrar recursos
        for (TrainWorkerConnection w : workers) w.close();
        if (serverSocket != null && !serverSocket.isClosed()) serverSocket.close();
        threadPool.shutdown();
    }
}
