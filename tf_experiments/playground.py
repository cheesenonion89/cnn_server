from cnn_server.server import file_service as dirs

d = dirs.get_transfer_proto_dir('cars', 9)

print(d)

print(dirs.get_bot_id_from_dir(d))
print(dirs.get_setting_id_from_dir(d))