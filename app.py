import cv2
from ultralytics import YOLO

# Configuração e inicialização do modelo YOLO
# Utiliza o modelo YOLOv8 nano por ser leve e ideal para protótipos rápidos e deploys eficientes
print("Carregando o modelo YOLOv8...")
modelo = YOLO("yolov8n.pt")


def processar_camera_local():
    """Acessa a câmera local do dispositivo utilizando OpenCV, processa os frames

    com o YOLO e exibe o resultado em uma janela nativa.
    """
    # Inicializa a captura de vídeo utilizando a câmera padrão do sistema (índice 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera do dispositivo.")
        return

    print("Câmera aberta com sucesso. Pressione 'q' para sair.")

    # Loop contínuo para leitura e processamento de frames em tempo real
    while True:
        # Captura o frame atual da câmera
        sucesso, frame = cap.read()

        if not sucesso:
            print("Erro: Falha ao capturar o frame da câmera.")
            break

        # Executa a inferência do YOLO no frame capturado
        resultados = modelo(frame, verbose=False)

        # Desenha as caixas delimitadoras e rótulos diretamente no frame
        frame_anotado = resultados[0].plot()

        # Exibe o frame resultante em uma janela do sistema operacionall
        cv2.imshow("Scanner com YOLO (Local)", frame_anotado)

        # Interrompe o loop imediatamente se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Libera os recursos de hardware da câmera e fecha todas as janelas abertas
    cap.release()
    cv2.destroyAllWindows()
    print("Recursos liberados e aplicação encerrada.")


def main():
    """Função principal que serve como ponto de entrada para o script."""
    # Simula o acionamento do botão para abrir a câmera via terminal
    entrada = (
        input("Digite '1' e pressione Enter para abrir a câmera: ").strip()
    )

    if entrada == "1":
        processar_camera_local()
    else:
        print("Operação cancelada pelo usuário.")


if __name__ == "__main__":
    main()
