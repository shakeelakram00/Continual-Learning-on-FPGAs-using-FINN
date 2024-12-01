import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")#########TO SUPRESS THE WARNINGS FROM THE OUTPUT ###RESET TO DEFAUL ON LAST LINE OF THE CODE
warnings.filterwarnings("ignore", category=UserWarning, module="OpenMP")
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'   ######TO SUPRESS "OpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option."
from npytodat_UP import make_weight_file
import json
import numpy as np

def quantized_to_integer(array, min_val, max_val, bit_width):
    # Map the Quantized range to the integer range [-7, 7] for 4 bit layers
    scaled_array = np.round(array * (2 ** bit_width / 2 - 1)).clip(min_val, max_val)
    return scaled_array

def WeightInterpreter(model, Accel_name):
    if Accel_name == "Accel7":
        with open('final_hw_config_Accel7.json', 'r') as file:
            config_data = json.load(file)
    elif Accel_name == "Accel6":
        with open('final_hw_config_Accel6.json', 'r') as file:
            config_data = json.load(file)
    elif Accel_name == "Accel5":
        with open('final_hw_config_Accel5.json', 'r') as file:
            config_data = json.load(file)

    j = 0 
    for i, (name, param) in enumerate(model.named_parameters()):
        dat_files_path = './runtime_weights/'
        EachLayer_npy_path = './npytodatState_dictnpy/'
        if i == 0 or i == 4 or i == 8 or i == 12 or i==16 or i==20:
            j = j+1
            param = param.detach().numpy()
            np.save(EachLayer_npy_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.npy', param)
                                    
        if i==0:
            CNV1npy = np.load(EachLayer_npy_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.npy')
            CNV1npyBip = CNV1npy.transpose(2, 3, 1, 0).reshape(-1, 21)
            if Accel_name == "Accel6" or Accel_name == "Accel7":
                CNV1npyBip = quantized_to_integer(CNV1npyBip, -7.0, 7.0, 4)
                bw = 4
                export_wdt = "INT4"
            elif Accel_name == "Accel5":
                CNV1npyBip[CNV1npyBip < 0] = -1
                CNV1npyBip[CNV1npyBip > 0] = 1
                bw = 1
                export_wdt = "BIPOLAR"
            pe = config_data.get(f"MatrixVectorActivation_0", {}).get("PE", None)
            simd = config_data.get(f"MatrixVectorActivation_0", {}).get("SIMD", None)
            mw, mh = CNV1npyBip.shape
            make_weight_file(CNV1npyBip, "decoupled_runtime",
                             dat_files_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.dat',
                             mw, mh, pe, simd,bw, export_wdt)
        if i==4:
            CNV1npy = np.load(EachLayer_npy_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.npy')
            CNV1npyB = CNV1npy.transpose(2, 3, 1, 0).reshape(-1, 21)
            if Accel_name == "Accel5":
                CNV1npyB[CNV1npyB < 0] = -1
                CNV1npyB[CNV1npyB > 0] = 1
                bw = 1
                export_wdt = "BIPOLAR"
            elif Accel_name == "Accel6" or Accel_name == "Accel7":
                CNV1npyB = quantized_to_integer(CNV1npyB, -7.0, 7.0, 4)
                bw = 4
                export_wdt = "INT4"
            pe = config_data.get("MatrixVectorActivation_1", {}).get("PE", None)
            simd = config_data.get("MatrixVectorActivation_1", {}).get("SIMD", None)
            mw, mh = CNV1npyB.shape
            make_weight_file(CNV1npyB, "decoupled_runtime",
                             dat_files_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.dat',
                             mw, mh, pe, simd,bw,export_wdt)

        if i==8:
            CNV1npy = np.load(EachLayer_npy_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.npy')
            CNV1npyB = CNV1npy.transpose(2, 3, 1, 0).reshape(-1, 21)
            if Accel_name == "Accel5":
                CNV1npyB[CNV1npyB < 0] = -1
                CNV1npyB[CNV1npyB > 0] = 1
                bw =1
                export_wdt = "BIPOLAR"
            elif Accel_name == "Accel6" or Accel_name == "Accel7":
                CNV1npyB = quantized_to_integer(CNV1npyB, -7.0, 7.0, 4)
                bw = 4
                export_wdt = "INT4"
            pe = config_data.get("MatrixVectorActivation_2", {}).get("PE", None)
            simd = config_data.get("MatrixVectorActivation_2", {}).get("SIMD", None)
            mw, mh = CNV1npyB.shape
            make_weight_file(CNV1npyB, "decoupled_runtime",
                             dat_files_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.dat',
                             mw, mh, pe, simd, bw,export_wdt)
        if i == 12:
            Lin2npy = np.load(EachLayer_npy_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.npy')
            Lin2npyB = np.transpose(Lin2npy)
            if Accel_name == "Accel5" or Accel_name == "Accel6":
                Lin2npyB[Lin2npyB < 0] = -1
                Lin2npyB[Lin2npyB > 0] = 1
                bw =1
                export_wdt = "BIPOLAR"
            elif Accel_name == "Accel7":
                Lin2npyB = quantized_to_integer(Lin2npyB, -7.0, 7.0, 4)
                bw =4
                export_wdt = "INT4"
            pe = config_data.get("MatrixVectorActivation_3", {}).get("PE", None)
            simd = config_data.get("MatrixVectorActivation_3", {}).get("SIMD", None)
            mw, mh = Lin2npyB.shape
            make_weight_file(Lin2npyB, "decoupled_runtime",
                             dat_files_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.dat',
                             mw, mh, pe, simd,bw,export_wdt)

        if i==16:
            Lin2npy = np.load(EachLayer_npy_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.npy')
            Lin2npyB = np.transpose(Lin2npy)    
            if Accel_name == "Accel7":
                Lin2npyB = quantized_to_integer(Lin2npyB, -7.0, 7.0, 4)
                bw = 4
                export_wdt = "INT4"
            elif Accel_name == "Accel5" or Accel_name == "Accel6":
                Lin2npyB[Lin2npyB < 0] = 0
                Lin2npyB[Lin2npyB > 0] = 1
                bw =1
                export_wdt = "BINARY"
            pe = config_data.get("MatrixVectorActivation_4", {}).get("PE", None)
            simd = config_data.get("MatrixVectorActivation_4", {}).get("SIMD", None)
            mw, mh = Lin2npyB.shape
            make_weight_file(Lin2npyB, "decoupled_runtime",
                             dat_files_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.dat',
                             mw, mh, pe, simd,bw,export_wdt)
        if i==20:
            Lin2npy = np.load(EachLayer_npy_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.npy')
            Lin2npyB = np.transpose(Lin2npy)
            if Accel_name == "Accel7":
                Lin2npyB = quantized_to_integer(Lin2npyB, -7.0, 7.0, 4)
                bw = 4
                export_wdt = "INT4"
            elif Accel_name == "Accel5" or Accel_name == "Accel6":
                Lin2npyB[Lin2npyB < 0] = 0
                Lin2npyB[Lin2npyB > 0] = 1
                bw =1
                export_wdt = "BINARY"
            pe = config_data.get("MatrixVectorActivation_5", {}).get("PE", None)
            simd = config_data.get("MatrixVectorActivation_5", {}).get("SIMD", None)
            mw, mh = Lin2npyB.shape
            make_weight_file(Lin2npyB, "decoupled_runtime",
                             dat_files_path+f'{j}_0_StreamingDataflowPartition_{j}_MatrixVectorActivation_0.dat',
                             mw, mh, pe, simd, bw,export_wdt)
    print(f"{Accel_name} Weights updated:\n")
