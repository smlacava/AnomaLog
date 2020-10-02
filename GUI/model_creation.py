from tensorflow.keras.utils import get_file
from AnomaLog import AnomaLog

# Data loading
name_ds = 'kddcup.data_10_percent.gz'
link_ds = 'http://kdd.ics.uci.edu/databases/kddcup99/'

try:
    path = get_file(name_ds, origin=link_ds + name_ds)
except:
    print('Error downloading')
    raise

IDS = AnomaLog(model_type='RF', multiclass=False)

columns_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'outcome']
df = IDS.dataset_reader(path, columns_names)
x, y = IDS.compute_dataset(df, 'outcome', 'normal.')
print(x[0])
IDS.fit(x, y)
IDS.save('B_KDD_RF')

M_IDS = AnomaLog(model_type='RF', multiclass=True)
df = M_IDS.dataset_reader(path, columns_names)
x, y = M_IDS.compute_dataset(df, 'outcome', 'normal.')
M_IDS.fit(x, y)
M_IDS.save('M_KDD_RF')
