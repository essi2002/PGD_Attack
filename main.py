import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from train import train
from mnist_dataset import mnist_dataset
import torch.nn as nn
def pgd_attack(model,eps,alpha,iters,images,labels):
    images = images.to(torch.float32) 
    loss = nn.CrossEntropyLoss()
    original_image = images.data
    for iter in range(iters):
        images.requires_grad = True
        output = model(images)
        model.zero_grad()
        cost = loss(output,labels)
        cost.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_image,min=-eps,max=eps)
        images = torch.clamp(original_image + eta,min=0,max=1).detach()
    return images
def imshow(images, labels,path):
    image_array = images[0].detach().numpy()  # Detach the tensor from the computational graph
    label = labels[0]

    # Create a plot to display the image
    plt.imshow(image_array.squeeze(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

    # Save the image
    save_path = f"./{path}.png"
    plt.imsave(save_path, image_array.squeeze(), cmap='gray',dpi = 300)
    print(f"Image saved at: {save_path}")
def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(root='./datasets/', train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root='./datasets/', train=True, download=False, transform=transform)
    train_loader = DataLoader(dataset=mnist_dataset(train_data), batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=mnist_dataset(test_data), batch_size=64, shuffle=True)
   
    images, labels = next(iter(train_loader))
   
    imshow(images, labels,"normal")
    
    model = train(train_loader)
    perturb_image = pgd_attack(model,0.3,2/255,40,images,labels)
    imshow(perturb_image,labels,"adversial")
if __name__ == "__main__":
    main()