from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from os import path
from tkinter import filedialog
from AnomaLog import AnomaLog
import pandas as pd

analyzer = AnomaLog()
analyzer.load('B_KDD_RF')
bias = analyzer.norm_bias

window = Tk()
window.configure(bg='white')
window.wm_iconbitmap('favicon.ico')
window.title("AnomaLog")
window.geometry('435x225')

style = Style()
style.theme_use('default')
style.configure("TCheckbutton", background='white', pressed='red', selectcolor='white')
style.configure('TButton', height=15, font=('calibri', 10), foreground='#092A9C')
style.configure("blue.Horizontal.TProgressbar", background='#092A9C')

n_row = 0

lbl_bar = Label(window, text="_______________________            ", anchor=W, foreground='blue', background='white')
lbl_bar.grid(column=0, row=n_row)

n_row = 1

lbl_w = Label(window, text="            ", anchor=W, foreground='blue', background='white')
lbl_w.grid(column=0, row=n_row)

n_row = 2

lbl_info = Label(window, text="Analyze your records to find", anchor='w', background='white')
lbl_info.grid(column=0, row=n_row)
lbl_info2 = Label(window, text="anomalies.                                           ", anchor='w', background='white')
lbl_info2.grid(column=1, row=n_row)

n_row = 3

lbl_file = Label(window, text="                 ", anchor='w', justify=RIGHT, background='white')
lbl_file.grid(column=0, row=n_row)

n_row = 4

lbl_file = Label(window, text="Select  the  file                  ", anchor='w', justify=RIGHT, background='white')
lbl_file.grid(column=0, row=n_row)

txt_file = Entry(window, width=25)
txt_file.grid(column=1, row=n_row)
txt_file.focus()


def file_search_clicked():
    filename = filedialog.askopenfilename()
    txt_file.delete(0, 'end')
    txt_file.insert(0, filename)


btn_file = Button(window, text="Search", command=file_search_clicked)
btn_file.grid(column=2, row=n_row)

n_row = 5

lbl_bias = Label(window, text="Select  the  normal  bias   ", anchor=W, background='white')
lbl_bias.grid(column=0, row=n_row)
v_bias = StringVar()
txt_bias = Entry(window, width=25, textvariable=v_bias)
v_bias.set(str(bias))
txt_bias.grid(column=1, row=n_row)

n_row = 6

lbl_file = Label(window, text="                  ", anchor='w', justify=RIGHT, background='white')
lbl_file.grid(column=0, row=n_row)

n_row = 7

chk_state = BooleanVar()
chk_state.set(False)
chk = Checkbutton(window, text='Only  anomalous', var=chk_state, style='TCheckbutton')
chk.grid(column=0, row=n_row)

chk_multi = BooleanVar()
chk_multi.set(False)
chk_class = Checkbutton(window, text='Multiclass', var=chk_multi)
chk_class.grid(column=1, row=n_row)

def analyze_clicked():
    bias = v_bias.get()
    window.update_idletasks()
    filename = txt_file.get()
    window.update_idletasks()
    outfile = filename.split('.')
    window.update_idletasks()
    outfile = outfile[0] + 'results'
    bar['value'] = 20
    window.update_idletasks()
    try:
        bias = float(bias)
        bar['value'] = 25
        window.update_idletasks()
        anomalFLAG = chk_state.get()
        multiFLAG = chk_multi.get()
        bar['value'] = 30
        window.update_idletasks()
        analyzer = AnomaLog()
        if multiFLAG is False:
            analyzer.load('B_KDD_RF')
        else:
            analyzer.load('M_KDD_RF')
        fields = analyzer.fields
        if path.isfile(filename) is False:
            messagebox.showinfo('File not found', 'The file does not exist')
        else:
            try:
                bar['value'] = 35
                window.update_idletasks()
                raw_df = pd.read_csv(filename, header=None)
                raw_df = raw_df.drop(raw_df.columns[0], axis=1)
                print('Data loaded')
                bar['value'] = 40
                window.update_idletasks()
                df = analyzer.analysis(raw_df, norm_bias=bias, anomalousFLAG=anomalFLAG)
                print("Analysis complete")
                bar['value'] = 70
                window.update_idletasks()
                names = dict()
                for i in range(len(fields)):
                    names[df.columns[i]] = fields[i]
                df.rename(columns=names, inplace=True)
                bar['value'] = 90
                window.update_idletasks()
                print("File creation")
                df.to_csv(outfile + ".csv", index=False, header=True)
                bar['value'] = 100
                window.update_idletasks()
                messagebox.showinfo('Success', "The result are saved in " + outfile + '.zip')
            except:
                messagebox.showinfo('Error', 'System problem')
    except:
        messagebox.showinfo('Wrong normal bias value', 'The normal bias has to be a number')
    bar['value'] = 0
    window.update_idletasks()

n_row = 8

lbl_bar2 = Label(window, text="_______________________            ", anchor=W, foreground='blue', background='white')
lbl_bar2.grid(column=0, row=n_row)

n_row = 9

lbl_w2 = Label(window, text="           ", anchor=W, foreground='blue', background='white')
lbl_w2.grid(column=0, row=n_row)

n_row = 10

btn_analyze = Button(window, text="Analyze", command=analyze_clicked)
btn_analyze.grid(column=0, row=n_row)


bar = Progressbar(window, length=155, style='blue.Horizontal.TProgressbar')
bar['value'] = 0
bar.grid(column=1, row=n_row)

window.mainloop()