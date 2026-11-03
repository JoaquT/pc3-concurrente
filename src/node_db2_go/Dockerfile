FROM fedora:39

# Instalar Go en Fedora (cumpliendo la heterogeneidad Nodo BD2: Go / Fedora)
RUN dnf install -y golang git && dnf clean all

WORKDIR /app

COPY go.mod ./
COPY db2_node.go ./

# Descargar dependencias de Go y compilar el binario nativo
RUN go mod tidy && go build -o db2_node db2_node.go

CMD ["./db2_node"]
