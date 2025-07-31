import torch
import torch.nn as nn
import threading
from Tx_Rx import transfer

def split_model(full_model, layer):
    layers = list(full_model.children())

    head = nn.Sequential(*layers[:layer])
    tail = nn.Sequential(*layers[layer:])

    return head, tail


def make_split_inference(head_func, tail_func, head_device='cpu', tail_device='mps', edge=False, cloud=False):
    def run(input):

        if not cloud:
            # Evaluate the head model
            with torch.no_grad():
                head_output = head_func(input)

        # If doing edge computing, return head output directly
        if edge:
            return head_output
        
        tx_size = head_output.element_size() * head_output.nelement()

        # EDGE -> CLOUD
        received = transfer(head_output)

        # Evaluate tail model
        with torch.no_grad():
            tail_output = tail_func(received.to(tail_device))

        # CLOUD -> EDGE
        received = transfer(tail_output)


        return received.to(head_device), tx_size

    return run

# def make_split_inference(head, tail, head_device='cpu', tail_device='mps'):
#     head = head.to(head_device)
#     tail = tail.to(tail_device)
    
#     head.eval()
#     tail.eval()

#     def run(input):
#         # Function to catch and store received data
#         received = {}
#         def receive_and_store():
#             received["data"] = receive()

#         # Evaluate the head model
#         with torch.no_grad():
#             head_output = head(input)

#         # If doing edge computing, return head output directly
#         if len(list(tail.children())) == 0:
#             return head_output
        
#         # EDGE -> CLOUD
#         server_thread = threading.Thread(target=receive_and_store)
#         server_thread.start()

#         transmit(head_output)  # transmit

#         server_thread.join()

#         # Evaluate tail model
#         with torch.no_grad():
#             tail_output = tail(received["data"].to(tail_device))

#         # CLOUD -> EDGE
#         received["data"] = {}
#         server_thread = threading.Thread(target=receive_and_store)
#         server_thread.start()

#         transmit(tail_output)  # transmit

#         server_thread.join()


#         return received["data"].to(head_device)

#     return run