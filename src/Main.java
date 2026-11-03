import java.io.File;
import java.util.Scanner;

public class Main {
    private static Process pythonProcess = null;

    public static void main(String[] args) throws Exception {
        if (args.length > 0) {
            handleCli(args);
            return;
        }

        printBanner();
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Ingrese opción [0-9]: ");
            String input = scanner.hasNextLine() ? scanner.nextLine().trim() : "1";
            if (input.isEmpty()) input = "1";

            switch (input) {
                case "1":
                    startAll();
                    return;
                case "2":
                    server.Server.main(new String[0]);
                    return;
                case "3":
                    node_java.JavaNode.main(new String[]{"Nodo_Java_1", "localhost", "5001"});
                    return;
                case "4":
                    startPythonNode("Nodo_Python_1", "localhost", "5001");
                    return;
                case "5":
                    client.Client.main(new String[0]);
                    return;
                case "6":
                    client.StressTestClient.main(new String[]{"15", "40", "200"});
                    return;
                case "7":
                    client.BenchmarkEvaluator.main(new String[]{"100"});
                    return;
                case "8":
                    try {
                        utils.DistributedTrainer.main(new String[]{"600", "25"});
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    return;
                case "9":
                    showNetworkDiagnostics();
                    break;
                case "0":
                    System.out.println("Saliendo.");
                    System.exit(0);
                    break;
                default:
                    System.out.println("Opción no válida.");
            }
        }
    }

    private static void printBanner() {
        System.out.println("==========================================================");
        System.out.println("               OPEN AI CUBES - LANZADOR");
        System.out.println("==========================================================");
        System.out.println("Seleccione una opción de ejecución:");
        System.out.println("  [1] Iniciar Sistema Completo (Servidor + Nodos + Chat GUI) [Recomendado]");
        System.out.println("  [2] Iniciar Servidor Central");
        System.out.println("  [3] Iniciar Nodo de Cálculo Java");
        System.out.println("  [4] Iniciar Nodo de Cálculo Python");
        System.out.println("  [5] Iniciar Cliente Chat (Interfaz Gráfica)");
        System.out.println("  [6] Ejecutar Prueba de Saturación (600 transacciones)");
        System.out.println("  [7] Ejecutar Benchmark (Secuencial / Paralelo / Distribuido)");
        System.out.println("  [8] Entrenar Red Neuronal Distribuida");
        System.out.println("  [9] Diagnóstico de Red e IPs para Conexión Remota");
        System.out.println("  [0] Salir");
        System.out.println("==========================================================");
    }

    private static void handleCli(String[] args) throws Exception {
        String cmd = args[0].toLowerCase();
        switch (cmd) {
            case "all":
                startAll();
                break;
            case "server":
                server.Server.main(shiftArgs(args));
                break;
            case "node-java":
                node_java.JavaNode.main(shiftArgs(args));
                break;
            case "node-python":
                String[] pArgs = shiftArgs(args);
                String nId = pArgs.length > 0 ? pArgs[0] : "Nodo_Python_1";
                String host = pArgs.length > 1 ? pArgs[1] : "localhost";
                String port = pArgs.length > 2 ? pArgs[2] : "5001";
                startPythonNode(nId, host, port);
                break;
            case "client":
                client.Client.main(shiftArgs(args));
                break;
            case "stress":
                client.StressTestClient.main(shiftArgs(args));
                break;
            case "benchmark":
                client.BenchmarkEvaluator.main(shiftArgs(args));
                break;
            case "train":
                try {
                    utils.DistributedTrainer.main(shiftArgs(args));
                } catch (Exception e) {
                    e.printStackTrace();
                }
                break;
            case "ip":
                showNetworkDiagnostics();
                break;
            default:
                System.out.println("Comando desconocido: " + cmd);
                System.out.println("Opciones válidas: all, server, node-java, node-python, client, stress, benchmark, train, ip");
        }
    }

    private static String[] shiftArgs(String[] args) {
        if (args.length <= 1) return new String[0];
        String[] rest = new String[args.length - 1];
        System.arraycopy(args, 1, rest, 0, rest.length);
        return rest;
    }

    public static void startAll() {
        System.out.println("==========================================================");
        System.out.println("   INICIANDO CLÚSTER COMPLETO EN UN SOLO PROCESO");
        System.out.println("==========================================================");

        // 1. Iniciar Servidor Central en segundo plano
        System.out.println("[Launcher] 1/4 Iniciando Servidor Central...");
        Thread serverThread = new Thread(() -> server.Server.main(new String[0]), "ServerThread");
        serverThread.setDaemon(true);
        serverThread.start();

        try { Thread.sleep(1200); } catch (InterruptedException ignored) {}

        // 2. Iniciar Nodo Java en segundo plano
        System.out.println("[Launcher] 2/4 Conectando Nodo Java...");
        Thread javaNodeThread = new Thread(() -> {
            node_java.JavaNode jNode = new node_java.JavaNode("Nodo_Java_1", "localhost", 5001);
            jNode.start();
        }, "JavaNodeThread");
        javaNodeThread.setDaemon(true);
        javaNodeThread.start();

        try { Thread.sleep(600); } catch (InterruptedException ignored) {}

        // 3. Iniciar Nodo Python en segundo plano si existe intérprete
        System.out.println("[Launcher] 3/4 Conectando Nodo Python...");
        spawnPythonNodeProcess("Nodo_Python_1", "localhost", 5001);

        try { Thread.sleep(800); } catch (InterruptedException ignored) {}

        // Registrar shutdown hook para limpiar proceso Python si fue iniciado
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (pythonProcess != null && pythonProcess.isAlive()) {
                pythonProcess.destroy();
            }
        }));

        // 4. Iniciar Interfaz Gráfica del Cliente
        System.out.println("[Launcher] 4/4 Abriendo Interfaz Gráfica del Cliente...");
        System.out.println("==========================================================");
        client.Client.main(new String[0]);
    }

    private static void spawnPythonNodeProcess(String nodeId, String host, int port) {
        String[] pyCommands = {"python3", "python"};
        File script = new File("src/node_python/node.py");
        if (!script.exists()) {
            System.out.println("[Launcher] Aviso: No se encontró 'src/node_python/node.py'.");
            return;
        }

        for (String py : pyCommands) {
            try {
                ProcessBuilder pb = new ProcessBuilder(py, script.getPath(), nodeId, host, String.valueOf(port));
                pb.inheritIO();
                pythonProcess = pb.start();
                System.out.println("[Launcher] Nodo Python iniciado con '" + py + "'.");
                return;
            } catch (Exception ignored) {}
        }
        System.out.println("[Launcher] Aviso: No se encontró intérprete python3/python en PATH.");
    }

    private static void startPythonNode(String nodeId, String host, String port) {
        String[] pyCommands = {"python3", "python"};
        File script = new File("src/node_python/node.py");
        for (String py : pyCommands) {
            try {
                ProcessBuilder pb = new ProcessBuilder(py, script.getPath(), nodeId, host, port);
                pb.inheritIO();
                Process p = pb.start();
                p.waitFor();
                return;
            } catch (Exception ignored) {}
        }
        System.err.println("No se pudo iniciar Python. Asegúrese de que python3 o python esté instalado.");
    }

    private static void showNetworkDiagnostics() {
        System.out.println("==========================================================");
        System.out.println("          DIAGNÓSTICO DE RED PARA DEMO EN VIVO");
        System.out.println("==========================================================");
        java.util.List<String> ips = new java.util.ArrayList<>();
        try {
            java.util.Enumeration<java.net.NetworkInterface> interfaces = java.net.NetworkInterface.getNetworkInterfaces();
            while (interfaces.hasMoreElements()) {
                java.net.NetworkInterface iface = interfaces.nextElement();
                if (iface.isLoopback() || !iface.isUp()) continue;
                java.util.Enumeration<java.net.InetAddress> addrs = iface.getInetAddresses();
                while (addrs.hasMoreElements()) {
                    java.net.InetAddress addr = addrs.nextElement();
                    if (addr instanceof java.net.Inet4Address) {
                        ips.add(addr.getHostAddress());
                        System.out.printf("  • Interfaz: %-10s -> IP: %s%n", iface.getName(), addr.getHostAddress());
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("Error al obtener interfaces: " + e.getMessage());
        }

        String primaryIp = ips.isEmpty() ? "TU_IP_LOCAL" : ips.get(0);
        System.out.println("----------------------------------------------------------");
        System.out.println("PUERTOS DEL SERVIDOR CENTRAL:");
        System.out.println("  • Puerto Clientes (Chat / Estrés): 5000");
        System.out.println("  • Puerto Nodos de Cálculo:          5001");
        System.out.println("----------------------------------------------------------");
        System.out.println("COMANDOS PARA CONECTAR DESDE OTRAS MÁQUINAS EN LA RED:");
        System.out.println("1. Conectar Nodo Java (desde otra laptop):");
        System.out.printf("   java -cp bin node_java.JavaNode Nodo_Java_Remoto %s 5001%n", primaryIp);
        System.out.println("2. Conectar Nodo Python (desde otra laptop):");
        System.out.printf("   python3 src/node_python/node.py Nodo_Python_Remoto %s 5001%n", primaryIp);
        System.out.println("3. Conectar Cliente Chat GUI (desde otra laptop):");
        System.out.printf("   java -cp bin client.Client %s 5000%n", primaryIp);
        System.out.println("==========================================================");
    }
}
