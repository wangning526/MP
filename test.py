import vit_args
vitargs = vit_args.get_args()

u_path  = vitargs.u_data_path
v_path  = vitargs.v_data_path
t_path  = vitargs.t_data_path
mp_path = vitargs.m_data_path


batch_size = vitargs.batch_size
epochs = vitargs.epochs
learning_rate = vitargs.lr


print(batch_size)