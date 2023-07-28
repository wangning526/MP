import argparse

def get_args():
    parser = argparse.ArgumentParser(description='VIT for Microplastics')
    parser.add_argument('--input_size', type=int, default=(75,360))
    parser.add_argument('--output_size', type=int, default=75 * 360)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=15)
    parser.add_argument('--embed_dim', type=int, default=675)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--u_data_path', type=str, default="data\\U10M")
    parser.add_argument('--v_data_path', type=str, default="data\\V10M")
    parser.add_argument('--t_data_path', type=str, default="data\\TMPS")
    parser.add_argument('--m_data_path', type=str, default="data\\MP")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weights', type=str, default='',help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    return parser.parse_args()


