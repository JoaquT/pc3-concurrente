package server;

import java.awt.image.BufferedImage;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import javax.imageio.ImageIO;
import utils.ImageClassifierHelper;
import utils.SimpleNeuralNetwork;

public class Server {
    public static final int CLIENT_PORT = 5000;
    public static final int NODE_PORT = 5001;

    private static final List<WorkerNodeConnection> activeNodes = Collections.synchronizedList(new ArrayList<>());
    private static final AtomicInteger nodeIndex = new AtomicInteger(0);

    // Modelo maestro local en el servidor
    private static SimpleNeuralNetwork masterModel;

    public static synchronized List<WorkerNodeConnection> getActiveNodes() {
        return new ArrayList<>(activeNodes);
    }

    public static WorkerNodeConnection getNextNode() {
        synchronized (activeNodes) {
            if (activeNodes.isEmpty()) return null;
            int idx = Math.abs(nodeIndex.getAndIncrement()) % activeNodes.size();
            return activeNodes.get(idx);
        }
    }

    public static SimpleNeuralNetwork getMasterModel() {
        return masterModel;
    }

    public static void setMasterModel(SimpleNeuralNetwork model) {
        masterModel = model;
    }

    public static void main(String[] args) {
        System.out.println("==========================================================");
        System.out.println("            SERVIDOR CENTRAL DISTRIBUIDO                  ");
        System.out.println("==========================================================");

        // Cargar o inicializar el modelo
        File f = new File("modelo_animales.bin");
        if (f.exists()) {
            try {
                masterModel = SimpleNeuralNetwork.load(f.getPath());
                System.out.println("[Servidor] Modelo neuronal 'modelo_animales.bin' cargado.");
            } catch (Exception e) {
                System.err.println("[Servidor] Error al cargar modelo: " + e.getMessage());
                masterModel = new SimpleNeuralNetwork(ImageClassifierHelper.TOTAL_FEATURES, 128, 64, 3, 0.03);
            }
        } else {
            System.out.println("[Servidor] Inicializando nueva red neuronal (566 -> 128 -> 64 -> 3)...");
            masterModel = new SimpleNeuralNetwork(ImageClassifierHelper.TOTAL_FEATURES, 128, 64, 3, 0.03);
        }

        // 1. Hilo para escuchar NODOS de procesamiento (Java y Python)
        new Thread(() -> {
            try (ServerSocket nodeSocket = new ServerSocket(NODE_PORT)) {
                System.out.println("[Servidor] Escuchando NODOS de procesamiento en puerto " + NODE_PORT + "...");
                while (true) {
                    Socket socket = nodeSocket.accept();
                    new Thread(() -> handleNewNode(socket)).start();
                }
            } catch (IOException e) {
                System.err.println("[Servidor] Error en puerto de nodos: " + e.getMessage());
            }
        }, "NodeListenerThread").start();

        // 2. Escuchar CLIENTES en puerto CLIENT_PORT con ThreadPool
        ExecutorService clientExecutor = Executors.newCachedThreadPool();
        try (ServerSocket clientSocket = new ServerSocket(CLIENT_PORT)) {
            System.out.println("[Servidor] Escuchando CLIENTES (Chat / StressTest) en puerto " + CLIENT_PORT + "...");
            while (true) {
                Socket client = clientSocket.accept();
                clientExecutor.execute(new ClientHandler(client));
            }
        } catch (IOException e) {
            System.err.println("[Servidor] Error en puerto de clientes: " + e.getMessage());
        }
    }

    private static void handleNewNode(Socket socket) {
        try {
            WorkerNodeConnection node = new WorkerNodeConnection(socket);
            activeNodes.add(node);
            System.out.printf("[Servidor] + Nodo registrado: %s (%s) desde %s | Nodos activos en clúster: %d%n",
                    node.getNodeId(), node.getLanguage(), node.getRemoteAddress(), activeNodes.size());

            // Sincronizar automáticamente los pesos actuales con el nuevo nodo
            if (masterModel != null) {
                node.syncWeights(masterModel);
            }

            // Monitorear conexión viva
            node.waitForDisconnect();
        } catch (Exception e) {
            System.err.println("[Servidor] Falló conexión con nodo: " + e.getMessage());
        } finally {
            activeNodes.removeIf(n -> n.getSocket() == socket);
            System.out.printf("[Servidor] - Nodo desconectado | Nodos activos en clúster: %d%n", activeNodes.size());
        }
    }

    /**
     * Sincroniza los pesos del modelo maestro con todos los nodos conectados.
     */
    public static synchronized void broadcastWeights(SimpleNeuralNetwork model) {
        masterModel = model;
        List<WorkerNodeConnection> nodes = getActiveNodes();
        for (WorkerNodeConnection node : nodes) {
            try {
                node.syncWeights(model);
            } catch (Exception e) {
                System.err.println("[Servidor] Error sincronizando pesos con " + node.getNodeId() + ": " + e.getMessage());
            }
        }
    }
}


class WorkerNodeConnection {
    private final Socket socket;
    private final DataInputStream in;
    private final DataOutputStream out;
    private final String nodeId;
    private final String language;
    private final String capabilities;
    private final String remoteAddress;

    public static class FullResult {
        public final String nodeId;
        public final String language;
        public final String bestClass;
        public final double confidence;
        public final double convMs;
        public final double mlpMs;

        public FullResult(String nodeId, String language, String bestClass, double confidence, double convMs, double mlpMs) {
            this.nodeId = nodeId;
            this.language = language;
            this.bestClass = bestClass;
            this.confidence = confidence;
            this.convMs = convMs;
            this.mlpMs = mlpMs;
        }
    }

    public static class ConvResult {
        public final String nodeId;
        public final double[] features;
        public final double elapsedMs;

        public ConvResult(String nodeId, double[] features, double elapsedMs) {
            this.nodeId = nodeId;
            this.features = features;
            this.elapsedMs = elapsedMs;
        }
    }

    public static class MlpResult {
        public final String nodeId;
        public final String bestClass;
        public final double confidence;
        public final double[] probabilities;
        public final double elapsedMs;

        public MlpResult(String nodeId, String bestClass, double confidence, double[] probabilities, double elapsedMs) {
            this.nodeId = nodeId;
            this.bestClass = bestClass;
            this.confidence = confidence;
            this.probabilities = probabilities;
            this.elapsedMs = elapsedMs;
        }
    }

    public WorkerNodeConnection(Socket socket) throws IOException {
        this.socket = socket;
        this.remoteAddress = (socket.getInetAddress() != null) ? socket.getInetAddress().getHostAddress() : "desconocido";
        this.in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
        this.out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));

        // Lectura de registro del nodo: "REGISTER:<nodeId>:<lang>:<caps>"
        String regLine = in.readUTF();
        String[] parts = regLine.split(":");
        if (parts.length >= 4) {
            this.nodeId = parts[1];
            this.language = parts[2];
            this.capabilities = parts[3];
        } else {
            this.nodeId = "Nodo_Desconocido";
            this.language = "DESCONOCIDO";
            this.capabilities = "CONV,MLP";
        }
    }

    public Socket getSocket() { return socket; }
    public String getNodeId() { return nodeId; }
    public String getLanguage() { return language; }
    public String getCapabilities() { return capabilities; }
    public String getRemoteAddress() { return remoteAddress; }

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

        String resp = in.readUTF();
        if (!resp.equals("SYNC_OK")) {
            System.err.println("[" + nodeId + "] Respuesta inesperada al sincronizar: " + resp);
        }
    }

    public synchronized FullResult executeClassify(double[] flatGray, double[] colorHist) throws IOException {
        out.writeUTF("TASK_CLASSIFY");
        out.writeInt(flatGray.length);
        for (double v : flatGray) out.writeDouble(v);
        out.writeInt(colorHist.length);
        for (double v : colorHist) out.writeDouble(v);
        out.flush();

        String resHeader = in.readUTF(); // "RESULT_CLASSIFY:<nodeId>:<className>:<conf>:<convMs>:<mlpMs>"
        String[] parts = resHeader.split(":");
        String resNodeId = parts.length >= 2 ? parts[1] : nodeId;
        String className = parts.length >= 3 ? parts[2] : "DESCONOCIDO";
        double conf = parts.length >= 4 ? Double.parseDouble(parts[3].replace(",", ".")) : 0.0;
        double convMs = parts.length >= 5 ? Double.parseDouble(parts[4].replace(",", ".")) : 0.0;
        double mlpMs = parts.length >= 6 ? Double.parseDouble(parts[5].replace(",", ".")) : 0.0;

        int probLen = in.readInt();
        double[] probs = new double[probLen];
        for (int i = 0; i < probLen; i++) probs[i] = in.readDouble();

        return new FullResult(resNodeId, language, className, conf, convMs, mlpMs);
    }

    public synchronized ConvResult executeConvChunk(double[] flatGray, double[] colorHist) throws IOException {
        out.writeUTF("TASK_CONV");
        out.writeInt(flatGray.length);
        for (double v : flatGray) out.writeDouble(v);
        out.writeInt(colorHist.length);
        for (double v : colorHist) out.writeDouble(v);
        out.flush();

        String resHeader = in.readUTF(); // "RESULT_CONV:<nodeId>:<elapsedMs>"
        String[] parts = resHeader.split(":");
        double elapsedMs = parts.length >= 3 ? Double.parseDouble(parts[2].replace(",", ".")) : 0.0;

        int featLen = in.readInt();
        double[] features = new double[featLen];
        for (int i = 0; i < featLen; i++) features[i] = in.readDouble();

        return new ConvResult(nodeId, features, elapsedMs);
    }

    public synchronized MlpResult executeMlpChunk(double[] features) throws IOException {
        out.writeUTF("TASK_MLP");
        out.writeInt(features.length);
        for (double f : features) out.writeDouble(f);
        out.flush();

        String resHeader = in.readUTF(); // "RESULT_MLP:<nodeId>:<className>:<conf>:<elapsedMs>"
        String[] parts = resHeader.split(":");
        String className = parts.length >= 3 ? parts[2] : "DESCONOCIDO";
        double conf = parts.length >= 4 ? Double.parseDouble(parts[3].replace(",", ".")) : 0.0;
        double elapsedMs = parts.length >= 5 ? Double.parseDouble(parts[4].replace(",", ".")) : 0.0;

        int probLen = in.readInt();
        double[] probs = new double[probLen];
        for (int i = 0; i < probLen; i++) probs[i] = in.readDouble();

        return new MlpResult(nodeId, className, conf, probs, elapsedMs);
    }

    public synchronized SimpleNeuralNetwork.Gradients executeTrainBatch(
            List<double[]> batchX, List<Integer> batchY, SimpleNeuralNetwork model) throws IOException {

        out.writeUTF("TASK_TRAIN_BATCH");
        int batchSize = batchX.size();
        out.writeInt(batchSize);

        for (int i = 0; i < batchSize; i++) {
            out.writeInt(batchY.get(i));
            double[] x = batchX.get(i);
            out.writeInt(x.length);
            for (double v : x) out.writeDouble(v);
        }
        out.flush();

        String resHeader = in.readUTF(); // "RESULT_TRAIN_BATCH:<nodeId>:<count>:<elapsedMs>"
        int sampleCount = in.readInt();

        SimpleNeuralNetwork.Gradients grads = new SimpleNeuralNetwork.Gradients(
                model.inputSize, model.hidden1Size, model.hidden2Size, model.outputSize
        );
        grads.sampleCount = sampleCount;

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

    public void waitForDisconnect() {
        try {
            while (socket.isConnected() && !socket.isClosed()) {
                Thread.sleep(2000);
            }
        } catch (Exception ignored) {}
    }

    public void close() {
        try {
            out.writeUTF("QUIT");
            out.flush();
            socket.close();
        } catch (Exception ignored) {}
    }
}


class ClientHandler implements Runnable {
    private final Socket clientSocket;
    private static final String[] CLASSES = {"gato", "mariposa", "perro"};

    public ClientHandler(Socket clientSocket) {
        this.clientSocket = clientSocket;
    }

    @Override
    public void run() {
        long tStart = System.currentTimeMillis();
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(clientSocket.getInputStream()));
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(clientSocket.getOutputStream()))) {

            int length = in.readInt();
            if (length <= 0 || length > 50_000_000) {
                out.writeUTF("Error: Tamaño de imagen inválido (" + length + " bytes)");
                out.flush();
                return;
            }

            byte[] imageBytes = new byte[length];
            in.readFully(imageBytes);

            // 1. Decodificar imagen y extraer datos crudos usando SDK estándar
            BufferedImage img = ImageIO.read(new ByteArrayInputStream(imageBytes));
            if (img == null) {
                out.writeUTF("Error: Formato de imagen no soportado.");
                out.flush();
                return;
            }
            ImageClassifierHelper.RawImageData raw = ImageClassifierHelper.extractRawData(img);

            List<WorkerNodeConnection> nodes = Server.getActiveNodes();

            String bestClass;
            double confidence;
            String nodeInfo;
            double convMs = 0;
            double mlpMs = 0;

            if (!nodes.isEmpty()) {
                // EJECUCIÓN CONCURRENTE / PARALELA EN TODOS LOS N NODOS ACTIVOS (JAVA, PYTHON, ETC.)
                int numNodes = nodes.size();
                java.util.concurrent.ExecutorService exec = java.util.concurrent.Executors.newFixedThreadPool(numNodes);
                List<java.util.concurrent.Future<WorkerNodeConnection.FullResult>> futures = new ArrayList<>();

                for (WorkerNodeConnection n : nodes) {
                    futures.add(exec.submit(() -> n.executeClassify(raw.flatGray, raw.colorHist)));
                }

                StringBuilder summary = new StringBuilder();
                String topClass = "DESCONOCIDO";
                double topConf = -1;
                double maxConv = 0;
                double maxMlp = 0;

                for (int i = 0; i < futures.size(); i++) {
                    WorkerNodeConnection node = nodes.get(i);
                    try {
                        WorkerNodeConnection.FullResult res = futures.get(i).get(5, java.util.concurrent.TimeUnit.SECONDS);
                        maxConv = Math.max(maxConv, res.convMs);
                        maxMlp = Math.max(maxMlp, res.mlpMs);

                        if (res.confidence > topConf) {
                            topConf = res.confidence;
                            topClass = res.bestClass;
                        }

                        summary.append(String.format("• Nodo %s (%s): clasificó como <b>%s</b> (%.1f%%) &nbsp;[Conv: %.1fms, MLP: %.1fms]<br>",
                                res.language, res.nodeId, res.bestClass.toLowerCase(), res.confidence, res.convMs, res.mlpMs));
                    } catch (Exception ex) {
                        summary.append(String.format("• Nodo %s (%s): error de conexión<br>", node.getLanguage(), node.getNodeId()));
                    }
                }
                exec.shutdown();

                bestClass = topClass;
                confidence = topConf;
                nodeInfo = summary.toString();
                convMs = maxConv;
                mlpMs = maxMlp;

            } else {
                // Modo Fallback Local
                long t0 = System.nanoTime();
                double[][] gray2D = new double[32][32];
                for (int y = 0; y < 32; y++) System.arraycopy(raw.flatGray, y * 32, gray2D[y], 0, 32);
                double[] convFeat = ImageClassifierHelper.computeConvolutions(gray2D);
                double[] fullFeat = new double[raw.colorHist.length + convFeat.length];
                System.arraycopy(raw.colorHist, 0, fullFeat, 0, raw.colorHist.length);
                System.arraycopy(convFeat, 0, fullFeat, raw.colorHist.length, convFeat.length);
                convMs = (System.nanoTime() - t0) / 1_000_000.0;

                long t1 = System.nanoTime();
                SimpleNeuralNetwork model = Server.getMasterModel();
                double[] probs = model.predict(fullFeat);
                int best = 0;
                for (int i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
                mlpMs = (System.nanoTime() - t1) / 1_000_000.0;
                bestClass = CLASSES[best].toUpperCase();
                confidence = probs[best] * 100.0;
                nodeInfo = "Servidor_Local (Java)";
            }

            long totalMs = System.currentTimeMillis() - tStart;

            // Formato estructurado: "CLASE|CONFIDENCIA|NODO_INFO|CONV_MS|MLP_MS|TOTAL_MS"
            String response = String.format("%s|%.2f|%s|%.2f|%.2f|%d",
                    bestClass, confidence, nodeInfo, convMs, mlpMs, totalMs);

            out.writeUTF(response);
            out.flush();

        } catch (Exception e) {
            System.err.println("[Servidor ClientHandler] Error procesando petición: " + e.getMessage());
        }
    }
}
