from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


def transform_data_to_train_test():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)  # Set num_workers=0

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)  # Set num_workers=0

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Save the DataLoaders to disk
    torch.save(trainloader, '/tmp/trainloader.pth')
    torch.save(testloader, '/tmp/testloader.pth')

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        'trainloader_path': '/tmp/trainloader.pth',
        'testloader_path': '/tmp/testloader.pth',
        'device': device_str,
        'classes': classes
    }




import os

def work_with_model(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='transform_data_to_train_test_task')

    # Convert the device string back to a torch.device object
    device = torch.device(data['device'])

    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net = net.to(device)

    model_path = '/tmp/model.pth'
    torch.save(net.state_dict(), model_path)

    return {
        'model_path': model_path,
        'device': data['device'],  # Keep passing the string representation of device
        'trainloader_path': data['trainloader_path']
    }




def criterion_optimizer(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='work_with_model')

    model_path = data['model_path']
    device = torch.device(data['device'])

    # Load the model from disk
    net = models.resnet50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)

    # Create the criterion (no need to pass it via XCom)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Save optimizer state and model back to disk
    optimizer_path = '/tmp/optimizer.pth'
    torch.save(optimizer.state_dict(), optimizer_path)

    return {
        'model_path': model_path,
        'optimizer_path': optimizer_path  # Do not pass the criterion in XCom
    }



def train(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='criterion_optimizer')

    model_path = data['model_path']
    optimizer_path = data['optimizer_path']

    # Recreate the criterion (CrossEntropyLoss is simple to instantiate)
    criterion = nn.CrossEntropyLoss()

    # Load the DataLoader from disk
    trainloader = torch.load(ti.xcom_pull(task_ids='work_with_model')['trainloader_path'])
    device = ti.xcom_pull(task_ids='work_with_model')['device']

    # Initialize the model
    net = models.resnet50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer.load_state_dict(torch.load(optimizer_path))

    def train_model(net, trainloader, criterion, optimizer, epochs=1):
        net.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')

    # Train the model
    train_model(net, trainloader, criterion, optimizer, epochs=1)

    # Save the trained model and optimizer state
    torch.save(net.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)


def test(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='work_with_model')

    model_path = data['model_path']
    device = data['device']

    # Load the model
    net = models.resnet50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)

    # Load the DataLoader from disk
    testloader = torch.load(ti.xcom_pull(task_ids='transform_data_to_train_test_task')['testloader_path'])

    def test_model(net, testloader):
        net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10,000 test images: {100 * correct / total:.2f}%')

    test_model(net, testloader)


dag = DAG(
    'welcome_dag',
    default_args={'start_date': days_ago(1)},
    schedule_interval='0 23 * * *',
    catchup=False
)

transform_data_to_train_test_task = PythonOperator(
    task_id='transform_data_to_train_test_task',
    python_callable=transform_data_to_train_test,
    dag=dag,
    provide_context=True
)

work_with_model_task = PythonOperator(
    task_id='work_with_model',
    python_callable=work_with_model,
    dag=dag,
    provide_context=True
)

criterion_optimizer_task = PythonOperator(
    task_id='criterion_optimizer',
    python_callable=criterion_optimizer,
    dag=dag,
    provide_context=True
)

train_task = PythonOperator(
    task_id='train',
    python_callable=train,
    dag=dag,
    provide_context=True
)

test_task = PythonOperator(
    task_id='test',
    python_callable=test,
    dag=dag,
    provide_context=True
)

# Set the dependencies between the tasks
transform_data_to_train_test_task >> work_with_model_task >> criterion_optimizer_task >> train_task >> test_task
