import echidna.core.spectra as spectra
import numpy
from scipy.integrate import quad


class Scale(object):
    """ A class for scaling the parameter space of a spectra.

    Attributes:
      _scale_factor (float): The factor you want to scale a parameter by.
    """

    def __init__(self):
        """ Initialise the Scale class.
        """
        self._scale_factor = 1.

    def _get_str_slice(self, ibin, axis, n_dim, n_bins):
        """ Creates a string representing a slice of the form
          e.g. [:, ibin:ibin+1, :]

        Args:
          ibin (int): Bin you want to slice.
          axis (int): Index of the dimension you are slice.
          n_dim (int): Number of dimensions in the spectrum.
          n_bins (int): The number of bins the dimension contains.
        """
        str_slice = "["
        for dim in range(n_dim):
            if dim == axis:
                if ibin < n_bins - 1:
                    str_slice += str(ibin) + ":" + str(ibin + 1) + ","
                else:
                    str_slice += str(ibin) + ":,"
            else:
                str_slice += ":,"
        return str_slice[:-1] + "]"

    def _get_str_slices(self, bin1, bin2, axis, n_dim, n_bins):
        """ Creates a string representing a slice of the form
          e.g. [:, bin1:bin2, :]

        Args:
          bin1 (int): Lower bin of slice.
          bin2 (int): Upper bin of slice.
          axis (int): Index of the dimension you are slice.
          n_dim (int): Number of dimensions in the spectrum.
          n_bins (int): The number of bins the dimension contains.
        """
        str_slice = "["
        for dim in range(n_dim):
            if dim == axis:
                if bin2 < n_bins - 1:
                    str_slice += str(bin1) + ":" + str(bin2) + ","
                else:
                    str_slice += str(bin1) + ":,"
            else:
                str_slice += ":,"
        return str_slice[:-1] + "]"

    def _fill(self, spectrum, spec_slice, scaled_spec, scale_slice, weight):
        """ Private function which fills slices from spectrum into a scaled
          spectrum with a given weight.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Original spectrum
            to scale.
          spec_slice (string): Slice(s) which will be filled from the original
            spectrum into the scaled spectrum.
          scaled_spec (:class:`echidna.core.spectra.Spectra`): Spectrum
            which scaled slices will be filled with.
          scale_slice (string): Slice which will be filled in the
            scaled spectrum.
          weight (float): Value to weight the slice with.
        """
        fill_cmd = ("scaled_spec._data" + scale_slice + " += spectrum._data" +
                    spec_slice + " * weight")
        exec(fill_cmd)

    def _one_bin_scale(self, spectrum, spec_bin, scaled_spec, scale_bin,
                       interp, x_low, x_high, step, dimension, axis, n_dim,
                       n_bins):
        """ Private function for scaling in the case where you require one
        bin from the original spectrum.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Original spectrum
            to scale.
          spec_bin (int): Nin of spectrum used for scaling
          scaled_spec (:class:`echidna.core.spectra.Spectra`): Spectrum you
            are scaling in.
          scale_bin (int): Bin of scaled_spec you are scaling in.
          interp (scipy.interolate.interp1d): Interpolation of the projection
            of the dimension in the original spectrum.
          x_low (float): Lower limit used in scaling from the original
            spectrum.
          x_high (float): Upper limit used in scaling from the original
            spectrum.
          step (float): Bin width of the original spectrum.
          dimension (string): Dimension you are scaling
          axis (int): Index of dimension in spectrum.
          n_dim (int): Number of dimensions in spectrum.
          n_bins (int): Number of bins of the dimension in the spectrum.
        """
        y = quad(interp, x_low, x_high)[0] / step
        if y <= 0:
            # Scaling to negative or no events. Skipping...
            return
        if numpy.isnan(y):
            # Interpolation failed. Use bin height
            y = spectrum.project(dimension)[spec_bin]
            if y == 0:
                return
        spec_slice = self._get_str_slice(spec_bin, axis, n_dim, n_bins)
        scale_slice = self._get_str_slice(scale_bin, axis, n_dim, n_bins)
        unscaled_sum = float(eval("spectrum._data" + spec_slice + ".sum()"))
        if unscaled_sum == 0:
            return
        weight = y / unscaled_sum
        self._fill(spectrum, spec_slice, scaled_spec, scale_slice, weight)

    def _two_bin_scale(self, spectrum, spec_bin_low, scaled_spec, scale_bin,
                       interp, x_low, x_low_edge, x_high, step, dimension,
                       axis, n_dim, n_bins):
        """ Private function for scaling in the case where you require two
        bins from the original spectrum.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Original spectrum
            to scale.
          spec_bin_low (int): Lower bin of spectrum used for scaling
          scaled_spec (:class:`echidna.core.spectra.Spectra`): Spectrum you
            are scaling in.
          scale_bin (int): Bin of scaled_spec you are scaling in.
          interp (scipy.interolate.interp1d): Interpolation of the projection
            of the dimension in the original spectrum.
          x_low (float): Lower limit used in scaling from the original
            spectrum.
          x_low_edge (float): Upper edge of the bin which contains x_low.
          x_high (float): Upper limit used in scaling from the original
            spectrum.
          step (float): Bin width of the original spectrum.
          dimension (string): Dimension you are scaling
          axis (int): Index of dimension in spectrum.
          n_dim (int): Number of dimensions in spectrum.
          n_bins (int): Number of bins of the dimension in the spectrum.
        """
        y = quad(interp, x_low, x_high)[0] / step
        if y <= 0:
            # Scaling to negative or no events. Skipping...
            return
        area_low = quad(interp, x_low, x_low_edge)[0]
        area_high = quad(interp, x_low_edge, x_high)[0]
        if numpy.isnan(area_low) or area_low <= 0.:
            # Interpolation failed. Use bin height
            area_low = (x_low_edge - x_low) *\
                spectrum.project(dimension)[spec_bin_low]
        if numpy.isnan(area_high) or area_high <= 0.:
            # Interpolation failed. Use bin height
            area_high = (x_high - x_low_edge) * \
                spectrum.project(dimension)[spec_bin_low + 1]
        if area_high <= 0. and area_low <= 0.:
            # Scaling to negative or no events. Skipping...
            return
        low_weight = area_low / (area_low + area_high)
        high_weight = area_high / (area_low + area_high)
        if numpy.isnan(y):
            # Interpolation failed. Use bin height
            y_low = spectrum.project(dimension)[spec_bin_low]
            y_high = spectrum.project(dimension)[spec_bin_low + 1]
            if y_low == 0 and y_high == 0:
                return
            elif y_low == 0:
                y = y_high
            elif y_high == 0:
                y = y_low
            else:
                y = 0.5 * (low_weight * y_low + high_weight * y_high)
        spec_slice_low = self._get_str_slice(spec_bin_low, axis, n_dim, n_bins)
        spec_slice_high = self._get_str_slice(spec_bin_low + 1, axis, n_dim,
                                              n_bins)
        scale_slice = self._get_str_slice(scale_bin, axis, n_dim, n_bins)
        unscaled_sum = float(eval("spectrum._data" + spec_slice_low +
                                  ".sum()"))
        if unscaled_sum != 0:
            weight = low_weight * y / unscaled_sum
            self._fill(spectrum, spec_slice_low, scaled_spec, scale_slice,
                       weight)
        unscaled_sum = float(eval("spectrum._data" + spec_slice_high +
                                  ".sum()"))
        if unscaled_sum != 0:
            weight = high_weight * y / unscaled_sum
            self._fill(spectrum, spec_slice_high, scaled_spec, scale_slice,
                       weight)

    def get_scale_factor(self):
        """ Returns the scale factor.

        Returns:
          float: The scale factor.
        """
        return self._scale_factor

    def set_scale_factor(self, scale_factor):
        """ Sets the scale factor.

        Args:
          scale_factor (float): Value you wish to set the scale factor to.

       Raises:
         ValueError: If scale_factor is zero or below.
        """
        if scale_factor <= 0.:
            raise ValueError("Scale factor must be positive and non-zero.")
        self._scale_factor = scale_factor

    def scale(self, spectrum, dimension, **kwargs):
        """ Scales a given spectrum's dimension.

        Args:
          spectrum (float): The spectrum you want to scale.
          dimension (string): The dimension of the spectrum you want to scale.
          kwargs (dict): To passed to the interpolation function in
            :class:`echidna.core.spectra.Spectra`

        Returns:
          :class:`echidna.core.spectra.Spectra`: The scaled spectrum.
        """
        prescale_sum = spectrum.sum()
        interp = spectrum.interpolate1d(dimension, **kwargs)
        sf = self.get_scale_factor()
        scaled_spec = spectra.Spectra(spectrum._name+"_sf" +
                                      str(sf),
                                      spectrum._num_decays,
                                      spectrum.get_config())
        n_dim = len(spectrum._data.shape)
        axis = spectrum.get_config().get_index(dimension)
        par = spectrum.get_config().get_par(dimension)
        low = par._low
        high = par._high
        n_bins = par._bins
        step = par.get_width()
        for ibin in range(n_bins):
            x_low = (par.get_bin_centre(ibin) - 0.5*step) / sf
            x_high = (par.get_bin_centre(ibin) + 0.5*step) / sf
            if x_low < low or x_high >= high:
                continue  # Trying to scale values outside range (Unknown)
            old_bin_low = par.get_bin(x_low)
            old_bin_high = par.get_bin(x_high)
            if old_bin_low == old_bin_high:
                self._one_bin_scale(spectrum, old_bin_low, scaled_spec, ibin,
                                    interp, x_low, x_high, step, dimension,
                                    axis, n_dim, n_bins)
                continue
            x_low_edge = par.get_bin_centre(old_bin_low) + 0.5*step
            if old_bin_low == old_bin_high - 1:
                # No complete bins
                self._two_bin_scale(spectrum, old_bin_low, scaled_spec, ibin,
                                    interp, x_low, x_low_edge, x_high, step,
                                    dimension, axis, n_dim, n_bins)
                continue
            x_high_edge = par.get_bin_centre(old_bin_high) - 0.5*step
            y = quad(interp, x_low, x_high)[0] / step
            area_low = quad(interp, x_low, x_low_edge)[0]
            area_high = quad(interp, x_high_edge, x_high)[0]
            if numpy.isnan(area_low) or area_low <= 0.:
                # Interpolation failed. Use bin height
                area_low = (x_low_edge - x_low) *\
                    spectrum.project(dimension)[old_bin_low]
            if numpy.isnan(area_high) or area_high <= 0.:
                # Interpolation failed. Use bin height
                area_high = (x_high - x_low_edge) *\
                    spectrum.project(dimension)[old_bin_high]
            if y <= 0. or numpy.isnan(y):
                y = 0
                if old_bin_low + 1 != old_bin_high - 1:
                    y += self.trapezium(spectrum, dimension, old_bin_low+1,
                                        old_bin_high-1, step)
                y += area_low + area_high
                if y <= 0.:
                    continue
            spec_slices = self._get_str_slices(old_bin_low, old_bin_high + 1,
                                               axis, n_dim, n_bins)
            scale_slice = self._get_str_slice(ibin, axis, n_dim, n_bins)
            area_tot = 0
            for jbin in range(old_bin_low+1, old_bin_high):
                bin_low = par.get_bin_centre(jbin) - 0.5 * step
                bin_high = par.get_bin_centre(jbin) + 0.5 * step
                area = quad(interp, bin_low, bin_high)[0]
                if area <= 0 or numpy.isnan(area):
                    # Interpolation failed. Use bin height
                    area = spectrum.project(dimension)[jbin]*(bin_high -
                                                              bin_low)
                area_tot += area
            area_tot += area_low + area_high
            for jbin in range(old_bin_low+1, old_bin_high):
                comp_slice = self._get_str_slice(jbin, axis, n_dim, n_bins)
                unscaled_sum = float(
                    eval("spectrum._data" + comp_slice + ".sum()"))
                if unscaled_sum == 0:
                    continue
                bin_low = par.get_bin_centre(jbin) - 0.5 * step
                bin_high = par.get_bin_centre(jbin) + 0.5 * step
                area = quad(interp, bin_low, bin_high)[0]
                if area <= 0 or numpy.isnan(area):
                    # Interpolation failed. Use bin height
                    area = spectrum.project(dimension)[jbin]*(bin_high -
                                                              bin_low)
                comp_slice = self._get_str_slice(jbin, axis, n_dim, n_bins)
                unscaled_sum = float(
                    eval("spectrum._data" + comp_slice + ".sum()"))
                if unscaled_sum == 0:
                    continue
                weight = (area / area_tot) * (y / unscaled_sum)
                self._fill(spectrum, comp_slice, scaled_spec, scale_slice,
                           weight)
            slice_low = self._get_str_slice(old_bin_low, axis, n_dim, n_bins)
            unscaled_sum = float(
                eval("spectrum._data" + slice_low + ".sum()"))
            if unscaled_sum != 0:
                low_weight = (area_low / area_tot) * (y / unscaled_sum)
                self._fill(spectrum, slice_low, scaled_spec, scale_slice,
                           low_weight)
            slice_high = self._get_str_slice(old_bin_high, axis, n_dim, n_bins)
            unscaled_sum = float(
                eval("spectrum._data" + slice_high + ".sum()"))
            if unscaled_sum != 0:
                high_weight = (area_high / area_tot) * (y / unscaled_sum)
                self._fill(spectrum, slice_high, scaled_spec, scale_slice,
                           high_weight)
        # renormalise to prescale number of counts
        scaled_spec._num_decays = scaled_spec.sum()
        scaled_spec.scale(prescale_sum)
        scaled_spec._num_decays = spectrum._num_decays
        return scaled_spec

    def trapezium(self, spectrum, dimension, bin_low, bin_high, step):
        """ Integrates via the trapezium rule. Used when quad or interpolation
          fails.

        Args:
        spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum you wish to
          integrate.
        dimension (string): Dimension you wish to integrate over.
        bin_low (int): Lower bound of integration.
        bin_high (int): Upper bound of integration.
        step (float): Bin width.

        Returns:
          float: Value of the integration.
        """
        y = 0
        for ibin in range(bin_low, bin_high):
            if ibin != bin_low:
                y += 2*spectrum.project(dimension)[ibin]
            else:
                y += spectrum.project(dimension)[ibin]
        y += spectrum.project(dimension)[bin_high]
        return 0.5 * step * y
