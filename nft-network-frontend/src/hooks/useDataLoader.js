import { useEffect, useState } from 'react';
import { fetchLocalGraph } from '../services/dataService';

export function useDataLoader(dataset = 'medium_density') {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    fetchLocalGraph(dataset)
      .then((json) => {
        if (!isMounted) return;
        setData(json);
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err);
      })
      .finally(() => {
        if (!isMounted) return;
        setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [dataset]);

  return { data, loading, error };
}


