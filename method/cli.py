from nerfbaselines.cli import main
from nerfbaselines import MethodSpec, register

main.add_lazy_command('method.train:train_command', 'train')

MethodSpec: MethodSpec = {
    "method_class": "method.nexus_splats:NexusSplats",
    "id": "nexus-splats",
}

register(MethodSpec)

main()