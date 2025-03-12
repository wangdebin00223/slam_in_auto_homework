template <int D, typename E>
void BaseMultiEdge<D, E>::linearizeOplus()
{
  const number_t delta = cst(1e-9);
  const number_t scalar = 1 / (2*delta);
  ErrorVector errorBak;
  ErrorVector errorBeforeNumeric = _error;

  dynamic_aligned_buffer<number_t> buffer{ 12 };

  for (size_t i = 0; i < _vertices.size(); ++i) {
    //Xi - estimate the jacobian numerically
    OptimizableGraph::Vertex* vi = static_cast<OptimizableGraph::Vertex*>(_vertices[i]);

    if (vi->fixed()) {
      continue;
    } else {
      internal::QuadraticFormLock lck(*vi);
      const int vi_dim = vi->dimension();
      assert(vi_dim >= 0);

      number_t* add_vi = buffer.request(vi_dim);

      std::fill(add_vi, add_vi + vi_dim, cst(0.0));
      assert(_dimension >= 0);
      assert(_jacobianOplus[i].rows() == _dimension && _jacobianOplus[i].cols() == vi_dim && "jacobian cache dimension does not match");
        _jacobianOplus[i].resize(_dimension, vi_dim);
      // add small step along the unit vector in each dimension
      for (int d = 0; d < vi_dim; ++d) {
        vi->push();
        add_vi[d] = delta;
        vi->oplus(add_vi);
        computeError();
        errorBak = _error;
        vi->pop();
        vi->push();
        add_vi[d] = -delta;
        vi->oplus(add_vi);
        computeError();
        errorBak -= _error;
        vi->pop();
        add_vi[d] = 0.0;

        _jacobianOplus[i].col(d) = scalar * errorBak;
      } // end dimension
    }
  }
  _error = errorBeforeNumeric;
}