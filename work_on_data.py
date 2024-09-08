from data_preprocess.preprocess_data import preprocess_bevs_and_waypoints, get_delta_waypoints_histogram, create_xy_bins


data_path = "./data/"
preprocess = False
play_delta_histogram = False
play_create_delta_bins = True


if preprocess:
    inp = input("Dont accidentally do sth stupid :).\nWant to continue? (y): ")
    if inp == "y":
        print("Resizing images and extracting delta waypoints begins...\n")
        preprocess_bevs_and_waypoints(data_path, (128, 128), verbose=True)
        print("\nResizing images and extracting delta waypoints done!")


if play_create_delta_bins:
    x_num_of_bins = 13
    y_num_of_bins = 13

    print("Creating bin limits...")
    min_delta, max_delta = -2, 2
    bin_lims_x, bin_lims_y = create_xy_bins(min_delta, max_delta, (x_num_of_bins, y_num_of_bins))
    print("X lims:")
    print(bin_lims_x, end="\n\n")
    print("Y lims:")
    print(bin_lims_y, end="\n\n")


if play_delta_histogram:
    print("Extracting the distribution of delta waypoints for x and y...\n")
    num_of_bins_for_plot = 100
    delta_x_values, delta_y_values = get_delta_waypoints_histogram(data_path, num_of_bins_for_plot,
                                                                   show_plots=True, save_plots=True,
                                                                   out_path="./play_data/")

